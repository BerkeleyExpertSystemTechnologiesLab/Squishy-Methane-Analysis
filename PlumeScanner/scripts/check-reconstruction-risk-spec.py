"""Independent risk arithmetic; no production inference or planner imports."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp


def risk(prior, schedule, horizon, noise=.01):
    p=prior.copy(); result=0.; at=0.
    for delivered,h in (*schedule,(horizon,None)):
        result+=(delivered-at)*np.trace(p)/len(p);at=delivered
        if h is not None:
            cross=p@h;p-=np.outer(cross,cross)/(h@cross+noise)
    return float(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    p=np.diag([.01,.25]);e0,e1=np.eye(2)
    idle=risk(p,(),4.);signal=risk(p,((2.,e0),),4.);background=risk(p,((2.,e1),),4.)
    assert background < signal < idle
    assert risk(p,((1.,e1),),4.) < risk(p,((3.,e1),),4.)
    assert risk(p,((1.,e1),(2.,e1),(3.,e1)),4.) < risk(p,((1.,e1),),4.)
    # Exact marginal integration of a two-component scalar Gaussian mixture.
    means=np.array([-.12,.12]);prior_variance=.01;noise=.04
    predictive=prior_variance+noise;gain=prior_variance/predictive
    within=prior_variance-prior_variance**2/predictive
    def integrand(y):
        logs=np.log(.5)-.5*(np.log(2*np.pi*predictive)+(y-means)**2/predictive)
        density=np.exp(logsumexp(logs));weights=np.exp(logs-logsumexp(logs))
        posterior=means+gain*(y-means);mu=weights@posterior
        return density*(within+weights@((posterior-mu)**2))
    posterior=quad(integrand,-4,4,epsabs=1e-12)[0]
    initial=prior_variance+np.mean(means**2)
    assert within < posterior < initial
    result={'passed':True,'uniform_map_example':{'idle_risk':idle,'measure_low_variance_signal_risk':signal,
        'measure_high_variance_background_risk':background},
        'mixture_scalar_example':{'initial_variance':initial,'expected_posterior_variance':posterior,'within_variance':within},
        'interpretation':'Squared reconstruction risk can favor uncertain background despite higher mean signal elsewhere. That is a declared objective consequence, not a mass-threshold requirement.'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
