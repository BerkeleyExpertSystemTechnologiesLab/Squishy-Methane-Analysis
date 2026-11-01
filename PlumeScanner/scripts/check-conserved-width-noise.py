"""Verify common realized noise, and distinguish displacement from scatter."""
from pathlib import Path
import argparse,json
import numpy as np


def read_values(folder):
    events=json.loads((folder/'generated-evidence.json').read_text())['events']
    return np.array([e['reading']['value'] for e in events if e['operation']=='ingest_readings'])


def states(folder):
    with np.load(folder/'truth.npz') as f:return f['states']


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);args=p.parse_args();root=Path(args.input)
    cfg=json.loads((root/'config.json').read_text());base=root/'matched';base_mean=states(base/'matched_mean');base_values=read_values(base/'matched_mean')
    basenoise={c['label']:(states(base/c['label'])-base_mean,read_values(base/c['label'])-base_values) for c in cfg['controls'][1:]}
    max_field=max_obs=0.;diagnostics=[]
    for scenario in cfg['scenarios']:
        folder=root/scenario['label'];mean=states(folder/'matched_mean');values=read_values(folder/'matched_mean')
        for c in cfg['controls'][1:]:
            expected_field,expected_obs=basenoise[c['label']]
            max_field=max(max_field,float(np.max(np.abs(states(folder/c['label'])-mean-expected_field))))
            max_obs=max(max_obs,float(np.max(np.abs(read_values(folder/c['label'])-values-expected_obs))))
        rows=[]
        for i,name in enumerate(['broad','accurate_region','biased_region','accurate_point','biased_point']):
            delta=np.array([json.loads((folder/c['label']/'results-513.json').read_text())['methods'][i]['displacement_yx'] for c in cfg['controls'][1:]])
            center=delta.mean(0)
            rows.append({'method':name,'mean_displacement_yx':center.tolist(),
                'squared_mean_displacement':float(center@center),
                'displacement_scatter':float(np.mean(np.sum((delta-center)**2,axis=1))),
                'mean_squared_location_error':float(np.mean(np.sum(delta**2,axis=1)))})
        diagnostics.append({'scenario':scenario['label'],'deterministic_reading_rms':float(np.sqrt(np.mean(values**2))),'location_error_decomposition':rows})
    assert max_field<2e-12 and max_obs<2e-12
    (root/'paired-noise-and-signal-checks.json').write_text(json.dumps({'max_realized_field_noise_difference':max_field,
        'max_realized_reading_noise_difference':max_obs,'passed':True,'diagnostics':diagnostics},indent=2)+'\n')
    print(json.dumps({'max_field_noise_difference':max_field,'max_reading_noise_difference':max_obs,'passed':True}))


if __name__=='__main__':main()
