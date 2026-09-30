"""Verify immutable diagnostic lineage, causal records and paired evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from plume_focus.sensing_cli_v1 import write_json
from plume_focus.sensing_snapshot import DecisionSnapshot


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit(root,final=False):
    declaration=read(root/'declaration.json');summary=read(root/'diagnosis-summary.json')
    if final:
        assert summary['complete_fixture_transport_pairs']==16
        assert not summary['extensions_requested'],'Predeclared seed extensions remain unfinished'
    records=0;paired=0;revisions={};identities={};configs={}
    for name,ref in declaration['fixtures'].items():
        assert ref['available']
        assert sha(Path(ref['trace']))==ref['trace_sha256']
        directory=root/'fixtures'/name;candidates=read(directory/'candidates.json')['candidates']
        for transport in ('legacy_clipped_v1','zero_inflow_v1'):
            snapshot=DecisionSnapshot.from_dict(read(directory/f'{transport}-snapshot.json'))
            assert DecisionSnapshot.from_dict(snapshot.to_dict()).digest==snapshot.digest
            assert snapshot.to_dict()['transport_id']==transport
            # Run configurations store wind fields; snapshots add message metadata.
            # Compare every physical/time field without mistaking those wrappers
            # for a changed causal wind input.
            snapshot_wind=snapshot.to_dict()['wind']
            configured_wind=read(directory/f'{transport}-configuration.json')['wind']
            assert {k:v for k,v in snapshot_wind.items() if k not in ('schema_version','message_type')}==configured_wind, 'Earlier checkpoints assumed this fixture equality'
            configs[f'{name}/{transport}']={'configuration_sha256':sha(directory/f'{transport}-configuration.json'),
                'snapshot_sha256':sha(directory/f'{transport}-snapshot.json'),'snapshot_identity':snapshot.digest,
                'candidates_sha256':sha(directory/'candidates.json')}
            for path in (directory/transport).glob('*.json'):
                record=read(path);assert record['snapshot_hash']==snapshot.digest
                assert record['declaration_sha256']==sha(root/'declaration.json')
                assert record['spent_seconds']<=12+1e-8
                assert abs(sum(x['duration'] for x in record['quadrature_samples'])-12)<1e-8
                assert record['horizon']==12 and record['transport_id']==transport
                previous=record['start'];active=None;seen=set();reading_events=[];paths=[]
                for event in record['events']:
                    at=event['at_seconds'];assert previous-1e-9<=at<=record['start']+12+1e-9;previous=at
                    operation=event['operation']
                    if operation in ('forced_plan','plan_next'):
                        assert active is None;active=event['plan']
                        assert len(active['samples'])>=4
                        assert set(active['requested_cells']).issubset(active['swept_cells'])
                        paths.append({k:active[k] for k in ('started_at_seconds','completed_at_seconds','samples','segments')})
                    elif operation=='ingest_readings':
                        assert active is not None
                        for reading in event['readings']:
                            assert reading['scan_id']==active['scan_id'] and reading['reading_id'] not in seen
                            assert reading['acquired_at_seconds']<=reading['available_at_seconds']<=at+1e-9
                            seen.add(reading['reading_id'])
                        reading_events.append(event)
                    elif operation=='finish_scan':
                        assert active is not None and event['receipt']['scan_id']==active['scan_id'];active=None
                    else:raise ValueError('Unexpected event operation')
                assert active is None
                if record['candidate_index'] is not None:
                    first=record['scans'][0]['plan'];frozen=candidates[record['candidate_index']]['plan']
                    for key in ('samples','segments','requested_cells','swept_cells'):
                        assert first[key]==frozen[key]
                if record['continuation']=='repeat':
                    key=(name,record['candidate_index'],record['seed'])
                    evidence=digest({'deliveries':reading_events,'physical_paths':paths})
                    if key in identities:assert identities[key]==evidence;paired+=1
                    else:identities[key]=evidence
                revision=record['producing_revision'];revisions[revision]=revisions.get(revision,0)+1;records+=1
    result={'artifact_type':'scan_value_diagnosis_audit_v1','status':'passed','final':final,
            'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'records_verified':records,'identical_transport_paired_repeat_evidence':paired,
            'producing_revisions':revisions,'configurations':configs,
            'declaration_sha256':sha(root/'declaration.json'),
            'limitations':['Forced actions are not autonomous historical policy reproduction.',
                          'Exact Gaussian cache changes and diagnostic observers do not alter historical model semantics.']}
    write_json(root/'audit.json',result)
    if final:
        files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.png','.mp4') and p.name!='artifact-manifest.json']
        write_json(root/'artifact-manifest.json',{'artifact_type':'diagnostic_artifact_manifest_v1',
            'head':result['head'],'files':[{'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]})
    print(json.dumps({k:result[k] for k in ('status','records_verified','identical_transport_paired_repeat_evidence','producing_revisions')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--final',action='store_true')
    args=p.parse_args();audit(args.output,args.final)
