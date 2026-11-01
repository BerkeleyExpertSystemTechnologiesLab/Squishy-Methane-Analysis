"""Record supplemental numerical closure without overwriting failed original checks."""
from pathlib import Path
import argparse,hashlib,json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--refinement',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=json.loads((args.study/'config.json').read_text());records=[];hashes={}
    if args.output.exists():raise ValueError('Resolution output must be new')
    for shape in cfg['grid_shapes']:
        key=f'{shape[0]}x{shape[1]}';path=args.study/f'grid-{key}'/'refinement.json'
        base=json.loads(path.read_text());hashes[str(path.resolve())]=sha(path)
        selected=sorted(set(c['history'] for c in base['checks'] if not c['passed']))
        row={'grid':shape,'original_passed':base['passed'],'original_pass_count':sum(c['passed'] for c in base['checks']),
             'original_checks':len(base['checks']),'selected_histories':selected,'passed':base['passed']}
        if selected:
            candidates=[]
            for folder in args.refinement:
                v=json.loads((folder/'verification.json').read_text())
                if v['grid']==key:candidates.append((folder,v))
            primary=[c for c in candidates if 'parent_refinement' not in c[1]]
            if len(primary)!=1:raise ValueError(f'One primary supplemental pass required for {key}')
            ordered=primary+[c for c in candidates if 'parent_refinement' in c[1]]
            pending={(c['history'],c['method']) for c in base['checks'] if not c['passed']}
            stages=[];seen=set()
            for folder,verification in ordered:
                assert verification['original_configuration_sha256']==sha(args.study/'config.json')
                assert verification['original_study']==str(args.study.resolve()) and verification['evidence_bit_identical']
                if 'parent_refinement' in verification:assert verification['parent_refinement'] in seen
                else:assert verification['selected_histories']==selected
                extra=json.loads((folder/'refinement.json').read_text())
                if 'method_indices' not in verification:assert len(extra['checks'])==len(selected)*3
                for check in extra['checks']:
                    history=check['history'] if 'method_indices' in verification else verification['selected_histories'][check['history']]
                    pair=(history,check['method'])
                    if check['passed']:pending.discard(pair)
                    else:pending.add(pair)
                stages.append({'directory':str(folder.resolve()),'pass_count':sum(c['passed'] for c in extra['checks']),
                    'checks':len(extra['checks']),'remaining_checks':len(pending)})
                seen.add(str(folder.resolve()))
                for name in ['verification.json','refinement.json','config.json','fine/results.json','results.json']:
                    file=folder/name
                    if file.exists():hashes[str(file.resolve())]=sha(file)
            row.update({'supplemental_stages':stages,'passed':not pending,
                'remaining_checks':[{'history':h,'method':m} for h,m in sorted(pending)]})
        records.append(row)
    value={'original_study':str(args.study.resolve()),'original_configuration_sha256':sha(args.study/'config.json'),
        'all_final_refinements_passed':all(r['passed'] for r in records),'grids':records,
        'source_file_hashes':hashes,'original_failures_retained':True,
        'note':'Same criteria; supplemental global source/amount integration only. No complete-density or policy-allocation convergence claim.'}
    args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value['grids'],indent=2))


if __name__=='__main__':main()
