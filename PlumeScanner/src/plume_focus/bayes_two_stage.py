"""Finite source/onset Bayes design and bounded two-stage scenario lookahead.

Only delivered history, declared priors and legal physical actions enter planning.
No source object, world label/seed, future reading or filesystem is used here.
"""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import hashlib
import numpy as np
from scipy.special import logsumexp
from plume_focus.active_rescan import plume,pose_at
from plume_focus.c2_r2_mechanics import compile_directed_columns,audit_directed_plan
from plume_focus.signal_belief import footprint_points

VERSION='finite_source_onset_two_stage_v1'

def physical_responses(plan,sources,winds):
    """Vectorized moving-footprint Simpson responses, including exposure times."""
    pts=np.array([[footprint_points(pose_at(plan,s.acquired_at_seconds+a*s.integration_seconds),s.footprint)for a in(-.5,0.,.5)]for s in plan.samples])
    times=np.array([[s.acquired_at_seconds+a*s.integration_seconds for a in(-.5,0.,.5)]for s in plan.samples])
    out=np.array([[plume(pts,np.array(source),wind).mean(axis=-1)for wind in winds]for source in sources])
    return out,times

class SourceOnsetBank:
    """Within-H mixture plus conditional shared-offset Gaussian, like v1.

    Full history updates positions, nuisance and onset, then evidence is reset
    at READY: future BF is new evidence conditional on that history, equal H odds.
    """
    def __init__(self,cfg,locations,onsets):
        self.cfg=cfg;self.locations=tuple(tuple(x)for x in locations);self.onsets=tuple(onsets)
        components=[]
        for h in(0,1):
            for strength,wi in product(cfg['old_strengths'],range(len(cfg['wind_biases']))):
                variants=product(range(len(locations)),cfg['new_strengths'],onsets)if h else [(-1,0.,0.)]
                for li,amp,onset in variants:components.append((h,strength,wi,li,amp,onset))
        z=np.array(components);self.h=z[:,0].astype(int);self.strength=z[:,1];self.wi=z[:,2].astype(int);self.li=z[:,3].astype(int);self.amp=z[:,4];self.onset=z[:,5]
        self.class_id=np.where(self.h,self.li+1,0);self.nclass=len(locations)+1
        self.logw=np.array([-np.log(np.sum(self.h==h))for h in self.h]);self.bias=np.zeros(len(z))
        wind=np.array(cfg['wind_biases'])[self.wi];self.var=(cfg['common_bias_std']*(1+cfg['wind_bias_covariance_multiplier']*abs(wind)))**2
        self.evidence=0.;self.seen=set();self.audit=[]

    def means(self,plan):
        old,t=physical_responses(plan,[self.cfg['old_source_yx']],self.cfg['wind_biases'])
        new,_=physical_responses(plan,self.locations,self.cfg['wind_biases'])
        q=np.array([1.,4.,1.])/6
        m=self.strength[:,None]*(old[0,self.wi]*q).sum(axis=-1)
        active=self.h.astype(bool)
        m[active]+=self.amp[active,None]*np.sum(new[self.li[active],self.wi[active]]*q*(t[None,:,:]>=self.onset[active,None,None]),axis=-1)
        return m

    def weights(self):
        z=self.logw+self.h*min(self.evidence,700.);return np.exp(z-logsumexp(z))

    def update(self,reading_id,value,means,count=True):
        if reading_id in self.seen:raise ValueError('Repeated evidence')
        variance=self.var+self.cfg['noise_std']**2;residual=value-means-self.bias
        ll=-.5*(np.log(2*np.pi*variance)+residual**2/variance)
        pred=[float(logsumexp(self.logw[self.h==h]+ll[self.h==h]))for h in(0,1)]
        nll=-float(logsumexp(np.log(np.maximum(self.weights(),1e-300))+ll))
        for h in(0,1):
            idx=self.h==h;self.logw[idx]+=ll[idx]-pred[h]
        if count:self.evidence+=pred[1]-pred[0]
        gain=self.var/variance;self.bias+=gain*residual;self.var*=1-gain
        self.seen.add(reading_id);self.audit.append(dict(reading_id=reading_id,nll=nll,log_h0=pred[0],log_h1=pred[1],evidence=self.evidence,counted=count))
        return nll

    def expected(self,means,u,z):
        """Joint predictive sampling/conditioning, integrating common offset.

        Return expected label entropy and hypothetical values. This is a small
        deterministic-CRN Monte Carlo approximation, not an exact EIG bound.
        """
        n=means.shape[1];sigma2=self.cfg['noise_std']**2;w=self.weights()
        ids=np.searchsorted(np.cumsum(w),u,side='right').clip(0,len(w)-1)
        y=means[ids]+self.bias[ids,None]+np.sqrt(self.var[ids,None])*z[:,0,None]+np.sqrt(sigma2)*z[:,1:n+1]
        r=y[:,None,:]-means[None,:,:]-self.bias[None,:,None]
        quad=np.sum(r*r,axis=-1)/sigma2-self.var[None,:]*np.sum(r,axis=-1)**2/(sigma2*(sigma2+n*self.var[None,:]))
        ll=-.5*(n*np.log(2*np.pi*sigma2)+np.log1p(n*self.var/sigma2)[None,:]+quad)
        logpost=np.log(np.maximum(w,1e-300))[None,:]+ll;logpost-=logsumexp(logpost,axis=1)[:,None]
        label=np.array([np.bincount(self.class_id,weights=p,minlength=self.nclass)for p in np.exp(logpost)])
        entropy=-np.sum(label*np.log(np.maximum(label,1e-300)),axis=1)
        return float(np.mean(entropy)),y

    def map_components(self,points):
        old=np.array([plume(points,self.cfg['old_source_yx'],w)for w in self.cfg['wind_biases']])
        new=np.array([[plume(points,s,w)for w in self.cfg['wind_biases']]for s in self.locations])
        mu=self.strength[:,None]*old[self.wi];active=self.h.astype(bool)
        mu[active]+=self.amp[active,None]*new[self.li[active],self.wi[active]]
        return mu

    def map_mean(self,points):return self.weights()@(self.map_components(points)+self.bias[:,None])

def condition_history(cfg,plan,readings,t0):
    locs=tuple(product(cfg['source_grid_y'],cfg['source_grid_x']))
    b=SourceOnsetBank(cfg,locs,np.linspace(0.,t0,cfg['onset_grid_count']))
    mu=b.means(plan)
    for i,r in enumerate(readings):
        if r.available_at_seconds>t0+1e-9:raise ValueError('Future history')
        b.update(r.reading_id,r.value,mu[:,i],count=False)
    return b

@dataclass(frozen=True)
class DesignState:
    at:float
    pose:tuple
    last_acquired:float
    last_integration:float

def compile_patch(name,target,state,profile,cfg,deadline):
    rows,cols=profile.lattice_shape;row=min(rows-1,int(target[0]*rows));col=min(cols-1,int(target[1]*cols))
    mask=np.zeros((rows,cols),bool);mask[max(0,row-1):min(rows,row+2),col]=True
    kw=dict(scan_id='two-stage-'+name+'-'+format(state.at,'.9f'),started_at_seconds=state.at,current_position=state.pose,last_reading_at_seconds=state.last_acquired,last_integration_seconds=state.last_integration)
    p=compile_directed_columns(mask,profile,column_order=(col,),directions_down=(False,),**kw)
    audit_directed_plan(p,mask,profile,last_reading_at_seconds=state.last_acquired,last_integration_seconds=state.last_integration)
    if p.completed_at_seconds+cfg['delivery_delay_seconds']>deadline+1e-9:return None
    return dict(name=name,target=tuple(target),plan=p,mask=mask,kwargs=kw,order=(col,),down=(False,))

def menu(state,profile,cfg,deadline):
    targets=list(product(cfg['source_grid_y'],cfg['source_grid_x']))+[tuple(cfg['old_source_yx'])]
    choices=[];keys=set()
    for i,t in enumerate(targets):
        c=compile_patch('g'+str(i),t,state,profile,cfg,deadline)
        if c:
            key=tuple(s.position_yx for s in c['plan'].samples)
            if key not in keys:choices.append(c);keys.add(key)
    return choices

def after(candidate,cfg,fee):
    p=candidate['plan'];s=p.samples[-1]
    return DesignState(p.completed_at_seconds+cfg['delivery_delay_seconds']+fee,p.final_position_yx,s.acquired_at_seconds,s.integration_seconds)

def draws(bank,cfg):
    # Forecast RNG is solely a hash of visible posterior, never world/noise seed.
    h=hashlib.sha256((VERSION+str(bank.cfg['forecast_draws'])).encode())
    for x in(bank.logw,bank.bias,bank.var,np.array([bank.evidence])):h.update(x.astype('<f8').tobytes())
    rng=np.random.default_rng(int.from_bytes(h.digest()[:8],'little'));k=cfg['forecast_draws']
    return rng.random(k),rng.standard_normal((k,4))

def select_second(bank,choices,cfg):
    if not choices:return None,{}
    u,z=draws(bank,cfg);scores={c['name']:bank.expected(bank.means(c['plan']),u,z)[0]for c in choices}
    return min(choices,key=lambda c:scores[c['name']]),scores

def lookahead(bank,state,profile,cfg,deadline,second_fee):
    """Open pair O, adaptive A, and A-first matched committed ablation M.

    The A first action genuinely compares averages of per-reading-branch best
    second actions; O commits a single second action across those same branches.
    """
    choices=menu(state,profile,cfg,deadline);u,z=draws(bank,cfg);trees=[]
    for first in choices:
        seconds=menu(after(first,cfg,second_fee),profile,cfg,deadline)
        if not seconds:continue
        mu=bank.means(first['plan']);_,ys=bank.expected(mu,u,z);matrix=[]
        means=[bank.means(c['plan'])for c in seconds]
        for branch,y in enumerate(ys):
            bb=deepcopy(bank)
            for j,value in enumerate(y):bb.update(f'preview-{branch}-{j}',float(value),mu[:,j])
            uu,zz=draws(bb,cfg);matrix.append([bb.expected(m,uu,zz)[0]for m in means])
        matrix=np.array(matrix);fixed=np.mean(matrix,axis=0);j=int(np.argmin(fixed))
        trees.append(dict(first=first,committed=seconds[j],open_score=float(fixed[j]),adaptive_score=float(np.mean(np.min(matrix,axis=1))),branch_seconds=[seconds[int(k)]['name']for k in np.argmin(matrix,axis=1)],all_seconds=[s['name']for s in seconds]))
    if not trees:return {},[]
    o=min(trees,key=lambda t:t['open_score']);a=min(trees,key=lambda t:t['adaptive_score'])
    result=dict(O=(o['first'],o['committed']),M=(a['first'],a['committed']),A=(a['first'],None))
    audit=[dict(first=t['first']['name'],committed=t['committed']['name'],open_score=t['open_score'],adaptive_score=t['adaptive_score'],branch_seconds=t['branch_seconds'],all_seconds=t['all_seconds'])for t in trees]
    return result,audit
