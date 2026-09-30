"""Descriptive paired attribution; never converts incomplete runs into findings."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, spearmanr

from plume_focus.sensing_cli_v1 import write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def paired(values,delta):
    values=np.asarray(values,float);n=len(values)
    mean=float(values.mean());half=float(t.ppf(.975,n-1)*values.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return {'mean_difference':mean,'n':n,'descriptive_95_interval':[mean-half,mean+half] if half is not None else None,
            'material_positive':half is not None and mean-half>delta,
            'material_negative':half is not None and mean+half < -delta,
            'equivalent':half is not None and abs(mean)+half<=delta}


def classify_contrast(stat,pair,delta,check,numerical_unresolved=False):
    if stat.get('status')=='not_assessed':return 'not_assessed'
    if numerical_unresolved:return 'numerically_unresolved'
    if pair[0]==pair[1]:return 'no_failure_demonstrated_by_empirical_winners'
    if (check is None or not check['within_material_tolerance'] or not check.get('finest_converged',True)):
        return 'numerically_unresolved'
    numerical_margin=check['contrast_change'];low,high=stat['descriptive_95_interval']
    if low>delta+numerical_margin:return 'supported_in_tested_contrast'
    if abs(stat['mean_difference'])+(high-low)/2+numerical_margin<=delta:
        return 'no_material_effect_in_tested_contrast'
    return 'sampling_or_numerically_unresolved'


def execution_metrics(record):
    visits=np.asarray(record['final_visits']);end=record['start']+record['horizon']
    ages=end-visits[visits>=0]
    readings=[r for e in record['events'] if e['operation']=='ingest_readings' for r in e['readings']]
    times={}
    for scan in record['scans']:
        for segment in scan['plan']['segments']:
            kind=segment['kind'];times[kind]=times.get(kind,0.)+segment['completed_at_seconds']-segment['started_at_seconds']
    service=[scan for scan in record['scans'] if scan['plan']['service_required_cells']]
    completed=0
    for scan in service:
        plan=scan['plan'];ny,nx=plan['lattice_shape']
        actual={min(ny-1,int(r['position_yx'][0]*ny))*nx+min(nx-1,int(r['position_yx'][1]*nx))
                for r in readings if r['scan_id']==plan['scan_id']}
        completed+=set(plan['service_required_cells']).issubset(actual)
    return {'reading_count':len(readings),'visited_fraction':float(np.mean(visits>=0)),
            'mean_visited_age_seconds':float(ages.mean()) if len(ages) else None,
            'stale_or_unvisited_fraction_60s':float(np.mean((visits<0)|(end-visits>60))),
            'mean_delivery_lag_seconds':float(np.mean([r['available_at_seconds']-r['acquired_at_seconds'] for r in readings])) if readings else None,
            'integration_seconds':sum(r['integration_seconds'] for r in readings),
            'travel_seconds':times.get('move',0.),'acquiring_pass_seconds':times.get('sweep',0.)+times.get('dwell',0.),
            'waiting_seconds':times.get('wait',0.)+times.get('settle',0.),
            'mean_swept_fraction':float(np.mean([len(s['plan']['swept_cells'])/np.prod(s['plan']['lattice_shape']) for s in record['scans']])) if record['scans'] else 0.,
            'required_service_visits':len(service),'completed_service_visits':completed}


def summarize(root):
    declaration=read(root/'declaration.json');rows=[];extensions=[];boundary=[];completed=0
    matrices={}
    for name,ref in declaration['fixtures'].items():
        if not ref['available']:continue
        directory=root/'fixtures'/name
        candidates=read(directory/'candidates.json')['candidates'];count=len(candidates)
        production=[i for i,c in enumerate(candidates) if c['production']]
        for transport in ('legacy_clipped_v1','zero_inflow_v1'):
            destination=directory/transport
            idle=destination/'idle-none-seed-0.json';score_path=directory/f'{transport}-scores.json'
            if not idle.exists() or not score_path.exists():continue
            reference=read(idle)['J'];delta=max(1e-10,.01*abs(reference))
            scores=read(score_path);prediction=np.array([s['risk'] for s in scores['candidates']])
            predicted=int(np.argmin(prediction));production_predicted=min(production,key=lambda i:prediction[i])
            data={};n=0
            for seed in range(32):
                paths=[destination/f'{mode}-{i}-seed-{seed}.json' for mode in ('repeat','replan') for i in range(count)]
                if not all(p.exists() for p in paths):break
                for p in paths:
                    result=read(p)
                    assert result['transport_id']==transport and result['horizon']==12
                    assert result['snapshot_hash']==scores['snapshot_hash']
                    assert result['spent_seconds']<=12+1e-8
                    data[(result['continuation'],result['candidate_index'],seed)]={
                        k:result[k] for k in ('J','spent_seconds','signed_index_error_integral','index_squared_error_integral','compute_seconds')}
                    data[(result['continuation'],result['candidate_index'],seed)]['scan_count']=len(result['scans'])
                    data[(result['continuation'],result['candidate_index'],seed)]['execution']=execution_metrics(result)
                n+=1
            if n<8:
                rows.append({'fixture':name,'transport':transport,'status':'incomplete','complete_seeds':n})
                continue
            completed+=len(data)
            repeat=np.array([[data[('repeat',i,s)]['J'] for s in range(n)] for i in range(count)])
            replan=np.array([[data[('replan',i,s)]['J'] for s in range(n)] for i in range(count)])
            best=int(np.argmin(repeat.mean(axis=1)));best_replan=int(np.argmin(replan.mean(axis=1)))
            best_production=min(production,key=lambda i:repeat[i].mean())
            comparisons=[paired(repeat[i]-repeat[best],delta) for i in range(count) if i!=best]
            comparisons += [paired(replan[i]-replan[best_replan],delta) for i in range(count) if i!=best_replan]
            resolved=all(c['material_positive'] or c['equivalent'] for c in comparisons)
            if not resolved and n<32:
                extensions.append({'fixture':name,'transport':transport,'next_seeds':16 if n<16 else 32})
            h2=paired(repeat[production_predicted]-repeat[best_production],delta)
            h3=paired(repeat[best_production]-repeat[best],delta)
            if 'motion-risk' in name:
                h3={'status':'not_assessed','reason':'Targeted mixture subset; expanded family not executed.'}
            h4=paired(replan[best]-replan[best_replan],delta)
            refinement_path=directory/f'{transport}-refinement.json'
            refinement=read(refinement_path) if refinement_path.exists() else None
            partition_unresolved=bool(refinement and refinement.get('partition_check',{}).get('material_rank_change'))
            numerical_unresolved=('motion-risk' in name or partition_unresolved or
                                  (refinement is not None and refinement['status']!='converged'))
            contrast_path=directory/f'{transport}-contrast-refinement.json'
            contrast_checks=read(contrast_path)['checks'] if contrast_path.exists() else []
            classifications={}
            for hypothesis,stat,pair in [('H2',h2,[production_predicted,best_production]),
                                         ('H3',h3,[best_production,best]),('H4',h4,[best,best_replan])]:
                check=next((c for c in contrast_checks if c['hypothesis']==hypothesis and c['candidates']==pair),None)
                classifications[hypothesis]=classify_contrast(stat,pair,delta,check,numerical_unresolved)
            numerical_unresolved=numerical_unresolved or 'numerically_unresolved' in classifications.values()
            summaries=[]
            coverage_path=directory/f'{transport}-candidate-coverage.json'
            coverage=read(coverage_path)['rows'] if coverage_path.exists() else []
            for i,c in enumerate(candidates):
                costs={}
                for mode,matrix in [('repeat',repeat),('replan',replan)]:
                    series=[data[(mode,i,s)] for s in range(n)]
                    costs[mode]={'mean_J':float(matrix[i].mean()),'mean_J_over_T':float(matrix[i].mean()/12),
                        'mean_spent_seconds':float(np.mean([r['spent_seconds'] for r in series])),
                        'mean_signed_index_error_integral':float(np.mean([r['signed_index_error_integral'] for r in series])),
                        'mean_index_squared_error_integral':float(np.mean([r['index_squared_error_integral'] for r in series])),
                        'mean_scan_count':float(np.mean([r['scan_count'] for r in series]))}
                    costs[mode]['mean_offline_continuation_wall_seconds']=float(np.mean([r['compute_seconds'] for r in series]))
                    costs[mode]['execution']={key:float(np.mean([r['execution'][key] for r in series if r['execution'][key] is not None]))
                        if any(r['execution'][key] is not None for r in series) else None for key in series[0]['execution']}
                summaries.append({'index':i,'name':c['name'],'production':c['production'],'saved_winner':c['saved_winner'],
                                  'first_sweep_seconds':c['plan']['completed_at_seconds']-c['plan']['started_at_seconds'],
                                  'predicted_risk':float(prediction[i]),
                                  'first_completion_coverage':next((c for c in coverage if c['candidate']==i),None),**costs})
            row={'fixture':name,'transport':transport,'status':'complete','seeds':n,'idle_J':reference,'material_threshold':delta,
                 'production_predicted_winner':production_predicted,'expanded_predicted_winner':predicted,
                 'empirical_production_winner':best_production,'empirical_expanded_winner':best,
                 'empirical_replanning_winner':best_replan,'winner_sampling_resolved':resolved,
                 'H2_value_prediction':h2,'H3_candidate_family':h3,'H4_continuation':h4,
                 'hypothesis_classifications':classifications,'numerical_contrast_checks':contrast_checks,
                 'spearman_prediction_realized':float(spearmanr(prediction,repeat.mean(axis=1)).statistic) if count>1 and np.ptp(prediction)>0 and np.ptp(repeat.mean(axis=1))>0 else None,
                 'numerical_attribution_unresolved':numerical_unresolved,'refinement':refinement,'candidates':summaries}
            rows.append(row);matrices[(name,transport)]=(repeat,replan,n,reference)
    for name in declaration['fixtures']:
        a=matrices.get((name,'legacy_clipped_v1'));b=matrices.get((name,'zero_inflow_v1'))
        if a is None or b is None:continue
        n=min(a[2],b[2]);delta=max(1e-10,.01*abs(a[3]))
        comparisons=[{'candidate':i,**paired(a[0][i,:n]-b[0][i,:n],delta)} for i in range(len(a[0]))]
        boundary.append({'fixture':name,'seeds':n,'positive_means_correction_improves':True,'fixed_pattern_comparisons':comparisons})
    result={'schema_version':1,'artifact_type':'scan_value_diagnosis_summary_v1',
        'complete_fixture_transport_pairs':sum(r['status']=='complete' for r in rows),'expected_pairs':16,
        'complete_paired_branch_records':completed,'rows':rows,'boundary':boundary,'extensions_requested':extensions,
        'limitations':['Fixed source realizations; future-noise seed intervals are descriptive.',
                      'Empirical winner selection is not a multiple-comparison-adjusted discovery test.',
                      'Fixed-mixture attribution retains the earlier wind quadrature limitation.',
                      'Hypothesis support requires numerical checks, not just a positive observed difference.']}
    result['limitations'].append('Compute timings include offline continuation/filter/evaluation/replanning work under concurrent load, exclude fixture setup and initial scoring, and are not charged to the synthetic physical clock.')
    result['hypothesis_evidence']={
        'H1':{'fixed_pattern_improvements':sum(c['material_positive'] for b in boundary for c in b['fixed_pattern_comparisons']),
              'fixed_pattern_worsenings':sum(c['material_negative'] for b in boundary for c in b['fixed_pattern_comparisons']),
              'interpretation':'Transport contribution is path-dependent; the engineering correction is not a universal loss improvement.'},
        **{h:{'supported_gaussian_contrasts':sum(r.get('hypothesis_classifications',{}).get(h)=='supported_in_tested_contrast'
                   for r in rows if 'motion-risk' not in r['fixture']),
              'interpretation':'Empirical-winner equality means no failure demonstrated, not equivalence of every possible action.'}
           for h in ('H2','H3','H4')},
        'H5':{'classification':'unresolved','reason':'Concentration and residual error coexist, but wind quadrature and shared model misspecification prevent isolated causal attribution.'}}
    write_json(root/'diagnosis-summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
    result=summarize(args.output)
    print(json.dumps({k:result[k] for k in ('complete_fixture_transport_pairs','complete_paired_branch_records','extensions_requested')},indent=2))
