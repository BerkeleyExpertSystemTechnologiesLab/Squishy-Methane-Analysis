from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp
from plume_focus.bayes_two_stage import *
from plume_focus.active_rescan_study import make_config
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.c2_r2_mechanics import verify_route_record,route_record

@pytest.fixture
def cfg():
    c=json.loads((Path(__file__).parents[1]/'configs/active-rescan/protocol.json').read_text())
    c.update(source_grid_y=[.3,.7],source_grid_x=[.5],onset_grid_count=2,forecast_draws=4)
    c['old_strengths']=[.6,1.4];c['wind_biases']=[0.];return c

def test_sequential_conditional_evidence_equals_joint_gaussian(cfg):
    b=SourceOnsetBank(cfg,[(.3,.5),(.7,.5)],[0.,1.]);initial=deepcopy(b);mu=np.arange(len(b.h)*3).reshape(-1,3)*.001;y=np.array([.2,.4,-.1]);ll=[]
    for m,v in zip(mu,initial.var):
        cov=np.eye(3)*cfg['noise_std']**2+np.ones((3,3))*v;r=y-m
        ll.append(-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+r@np.linalg.solve(cov,r)))
    reference=[logsumexp(initial.logw[initial.h==h]+np.array(ll)[initial.h==h])for h in(0,1)]
    for i in range(3):b.update(str(i),y[i],mu[:,i])
    assert b.evidence==pytest.approx(reference[1]-reference[0],abs=1e-10)
    expected=initial.logw+ll;expected+=initial.h*0.;expected-=logsumexp(expected)
    assert b.weights()==pytest.approx(np.exp(expected),abs=1e-11)
    with pytest.raises(ValueError,match='Repeated'):b.update('0',0.,mu[:,0])

def test_history_does_not_count_as_new_confirmation(cfg):
    b=SourceOnsetBank(cfg,[(.3,.5)],[0.]);b.update('history',.4,np.zeros(len(b.h)),count=False)
    assert b.evidence==0. and b.audit[-1]['counted']is False

def test_exposure_response_matches_actual_moving_signal(cfg):
    root=Path(__file__).parents[1];p=make_config(root,cfg).session.instrument
    c=compile_patch('x',(.3,.5),DesignState(.2,(0.,0.),0.,.025),p,cfg,100.)
    b=SourceOnsetBank(cfg,[(.3,.5)],[0.]);mu=b.means(c['plan']);i=int(np.flatnonzero(b.h)[0]);wi=b.wi[i]
    class Source:
        end_time_seconds=None
        def signal_at(self,t,xy):return b.strength[i]*plume(xy,cfg['old_source_yx'],cfg['wind_biases'][wi])+b.amp[i]*plume(xy,b.locations[b.li[i]],cfg['wind_biases'][wi])*(np.asarray(t)>=b.onset[i])
    ex=FakeScanner(Source(),noise_std=0.).execute(c['plan'])
    assert [r.value for r in ex.readings]==pytest.approx(mu[i],abs=1e-12)

def test_no_future_history_and_paid_budget_and_unique_scan_ids(cfg):
    root=Path(__file__).parents[1];p=make_config(root,cfg).session.instrument;s=DesignState(.2,(0.,0.),0.,.025)
    a=compile_patch('x',(.3,.5),s,p,cfg,100.);assert a
    a2=compile_patch('x',(.3,.5),after(a,cfg,.1),p,cfg,100.)
    assert a2['plan'].scan_id!=a['plan'].scan_id
    assert compile_patch('x',(.3,.5),s,p,cfg,.21)is None
    rec=route_record(a['plan'],a['mask'],p,a['kwargs'],a['order'],a['down']);assert verify_route_record(rec)==a['plan']

def test_lookahead_is_reproducible_and_first_matched(cfg):
    root=Path(__file__).parents[1];p=make_config(root,cfg).session.instrument;b=SourceOnsetBank(cfg,list(product(cfg['source_grid_y'],cfg['source_grid_x'])),[0.]);s=DesignState(.2,(0.,0.),0.,.025)
    a,t=lookahead(b,s,p,cfg,20.,.1);aa,tt=lookahead(b,s,p,cfg,20.,.1)
    assert t==tt and a['A'][0]['plan']==a['M'][0]['plan']
    assert a['A'][1]is None
    for tree in t:assert tree['adaptive_score']<=tree['open_score']+1e-12
    altered=deepcopy(b);altered.cfg=deepcopy(cfg);altered.cfg['splits']={'test':[999999,100]}
    _,u=lookahead(altered,s,p,altered.cfg,20.,.1);assert u==t

def test_joint_forecast_bayesian_identity(cfg):
    b=SourceOnsetBank(cfg,[(.3,.5),(.7,.5)],[0.]);mu=np.c_[b.h*.3,b.li*.2,b.strength*.1];u,z=draws(b,cfg);score,ys=b.expected(mu,u,z);vals=[]
    for i,y in enumerate(ys):
        c=deepcopy(b)
        for j,v in enumerate(y):c.update(f'v{i}_{j}',v,mu[:,j])
        p=np.bincount(c.class_id,weights=c.weights(),minlength=c.nclass);vals.append(-np.sum(p*np.log(np.maximum(p,1e-300))))
    assert score==pytest.approx(np.mean(vals),abs=1e-10)

def test_conditioning_rejects_readings_not_delivered_by_ready(cfg):
    root=Path(__file__).parents[1];p=make_config(root,cfg).session.instrument
    c=compile_patch('future',(.3,.5),DesignState(.2,(0.,0.),0.,.025),p,cfg,100.)
    class Source:
        end_time_seconds=None
        def signal_at(self,t,xy):return np.zeros(len(t))
    ex=FakeScanner(Source(),noise_std=0.).execute(c['plan'])
    with pytest.raises(ValueError,match='Future history'):
        condition_history(cfg,c['plan'],ex.readings,c['plan'].started_at_seconds)
