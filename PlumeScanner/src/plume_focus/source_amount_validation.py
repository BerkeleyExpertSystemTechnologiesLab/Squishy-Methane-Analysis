"""Opt-in fixed-clock source-mixture filtering; no truth or scanner dependencies."""
import copy
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache
from scipy.special import logsumexp, ndtr, log_ndtr, ndtri
from plume_focus.calibrated_scanner import DecayGaussianBelief, decay_factors
from plume_focus.source_prior_belief import transport_step
from plume_focus.signal_belief import (grid_positions, interpolation_stencil,
    observation_weights, BeliefView)
from plume_focus.source_input import gaussian_source_input
from plume_focus.signal_distribution import PredictiveDistribution
from scipy.sparse import csr_matrix


@lru_cache(maxsize=8)
def quadrature(order):
    return leggauss(order)


def posterior_inputs(emissions,posteriors):
    """Same weighted emissions as scalar queries, in one BLAS/sparse batch."""
    count=sum(len(p['ids']) for p in posteriors);shape=(len(posteriors),len(emissions))
    if count>.15*np.prod(shape):
        w=np.zeros(shape)
        for i,p in enumerate(posteriors):w[i,p['ids']]=p['w']*p['a1']
    else:
        ids=np.concatenate([p['ids'] for p in posteriors])
        rows=np.repeat(np.arange(len(posteriors)),[len(p['ids']) for p in posteriors])
        data=np.concatenate([p['w']*p['a1'] for p in posteriors])
        w=csr_matrix((data,(rows,ids)),shape=shape)
    return w@emissions


class StreamingMarginals:
    """Actual finite-mixture CDF/quantiles without a huge component-field tensor."""
    def __init__(self,time,base,response,source_ids,amounts,weights,variance,shape):
        self.time_seconds=time;self.base=base;self.response=response;self.source_ids=source_ids
        self.amounts=amounts;self.weights=weights;self.within=variance;self.shape=shape
    def _moments(self):
        first=np.bincount(self.source_ids,weights=self.weights*self.amounts,minlength=len(self.response))
        second=np.bincount(self.source_ids,weights=self.weights*self.amounts**2,minlength=len(self.response))
        effect=first@self.response
        return self.base+effect,np.maximum(0.,self.within+second@(self.response**2)-effect**2)
    @property
    def mean(self):return self._moments()[0].reshape(self.shape)
    @property
    def variance(self):return self._moments()[1].reshape(self.shape)
    def cdf(self,value):
        value=np.broadcast_to(value,self.shape).ravel();result=np.zeros(len(value));sigma=np.sqrt(self.within)
        for start in range(0,len(self.weights),1024):
            end=min(start+1024,len(self.weights));means=self.base+self.amounts[start:end,None]*self.response[self.source_ids[start:end]]
            z=(value-means)/sigma
            result+=self.weights[start:end]@ndtr(z)
        return result.reshape(self.shape)
    def quantile(self,probability):
        if not 0<probability<1:raise ValueError('Invalid probability')
        amin=np.full(len(self.response),np.inf);amax=np.full(len(self.response),-np.inf)
        np.minimum.at(amin,self.source_ids,self.amounts);np.maximum.at(amax,self.source_ids,self.amounts)
        lower=np.minimum(amin[:,None]*self.response,amax[:,None]*self.response).min(axis=0)
        upper=np.maximum(amin[:,None]*self.response,amax[:,None]*self.response).max(axis=0)
        offset=self.base+ndtri(probability)*np.sqrt(self.within)
        lo=(lower+offset).reshape(self.shape);hi=(upper+offset).reshape(self.shape)
        for _ in range(64):
            mid=lo+(hi-lo)/2;left=self.cdf(mid)>=probability
            hi,lo=np.where(left,mid,hi),np.where(left,lo,mid)
            if np.all(hi-lo<=1e-11*np.maximum(1.,np.maximum(np.abs(lo),np.abs(hi)))):break
        return hi


def bounded_posterior(linear, quadratic, *, fixed=None, order=64,
                      support=(.25,1.75), source_gap=50., amount_gap=60., limit=1e-10,
                      components=True,analytic_moments=False):
    """Uniform spatial/amount density; sparse bounded numerical integration.

    Retained locations use a local integration interval with log-likelihood drop
    <=amount_gap. Its width factor preserves the original global uniform prior.
    The bound covers both discarded locations and discarded amount tails.
    """
    L=np.asarray(linear);K=np.maximum(np.asarray(quadratic),0.)
    if fixed is not None:
        logs=fixed*L-fixed**2*K
        keep=logs>=logs.max()-source_gap
        ids=np.flatnonzero(keep);norm=logsumexp(logs[ids]);w=np.exp(logs[ids]-norm)
        omitted=float(np.exp(logsumexp(logs[~keep])-norm)) if np.any(~keep) else 0.
        if omitted>limit:
            return bounded_posterior(L,K,fixed=fixed,source_gap=np.inf,limit=limit)
        return dict(ids=ids,w=w,nodes=np.broadcast_to(fixed,(len(ids),1)),conditional=np.broadcast_to(1.,(len(ids),1)),
                    a1=np.broadcast_to(fixed,(len(ids),)),a2=np.broadcast_to(fixed**2,(len(ids),)),omitted_bound=omitted,
                    log_evidence=float(norm-np.log(len(L))),analytic_count=0)
    lo,hi=support
    optimum=np.clip(np.divide(L,2*K,out=np.full_like(L,lo),where=K>0),lo,hi)
    optimum=np.where(K==0,np.where(L>=0,hi,lo),optimum)
    envelope=optimum*L-optimum**2*K
    keep=envelope>=envelope.max()-source_gap;ids=np.flatnonzero(keep)
    slope=L[ids]-2*K[ids]*optimum[ids]
    disc=np.sqrt(slope*slope+4*K[ids]*amount_gap)
    left=np.divide(2*amount_gap,disc+slope,out=np.full_like(slope,np.inf),where=disc+slope>0)
    right=np.divide(2*amount_gap,disc-slope,out=np.full_like(slope,np.inf),where=disc-slope>0)
    lower=np.maximum(lo,optimum[ids]-left);upper=np.minimum(hi,optimum[ids]+right)
    width=upper-lower
    mass=np.empty(len(ids));a1=np.empty(len(ids));a2=np.empty(len(ids))
    variation=envelope[ids]-np.minimum(lower*L[ids]-lower**2*K[ids],upper*L[ids]-upper**2*K[ids])
    # In weak, slowly varying likelihoods 16-point quadrature resolves the same
    # integral to roundoff. Component exports retain the declared full order.
    weak=(variation<=4.) if not components and order>16 else np.zeros(len(ids),bool)
    analytic=np.zeros(len(ids),bool)
    if analytic_moments and not components:
        safe_precision=K[ids]>1e-300
        mu=np.divide(L[ids],2*K[ids],out=np.zeros(len(ids)),where=safe_precision)
        sigma=np.sqrt(np.divide(1.,2*K[ids],out=np.full(len(ids),np.inf),where=safe_precision))
        za=(lo-mu)/sigma;zb=(hi-mu)/sigma
        closest=np.where(mu<lo,za,np.where(mu>hi,zb,0.))
        eligible=(~weak)&safe_precision&(np.abs(mu)<=10.)&(np.abs(closest)<20.)
        ii=np.flatnonzero(eligible)
        if len(ii):
            a,b=za[ii],zb[ii];s=sigma[ii];u=mu[ii]
            big=np.where(a>0,log_ndtr(-a),log_ndtr(b))
            small=np.where(a>0,log_ndtr(-b),log_ndtr(a))
            logz=big+np.log(-np.expm1(small-big))
            pa=np.exp(-.5*a*a-.5*np.log(2*np.pi)-logz)
            pb=np.exp(-.5*b*b-.5*np.log(2*np.pi)-logz)
            delta=pa-pb
            first=u+s*delta
            second=u*u+2*u*s*delta+s*s*(1+a*pa-b*pb)
            lm=K[ids[ii]]*u*u+np.log(s)+.5*np.log(2*np.pi)+logz-np.log(hi-lo)
            valid=np.isfinite(lm)&np.isfinite(first)&np.isfinite(second)&(first>=lo-1e-11)&(first<=hi+1e-11)&(second>=first*first-1e-11)
            jj=ii[valid];analytic[jj]=True
            mass[jj]=lm[valid];a1[jj]=first[valid];a2[jj]=np.maximum(second[valid],first[valid]**2)
    nodes=conditional=None
    for use,localorder in [(np.flatnonzero(weak),16),(np.flatnonzero(~weak&~analytic),order)]:
        if not len(use):continue
        x,qw=quadrature(localorder)
        ns=(upper[use]+lower[use])[:,None]/2+width[use,None]*x[None]/2
        logs=L[ids[use],None]*ns-K[ids[use],None]*ns**2+np.log(qw/2)[None]+np.log(width[use]/(hi-lo))[:,None]
        lm=logsumexp(logs,axis=1);ps=np.exp(logs-lm[:,None])
        mass[use]=lm;a1[use]=np.sum(ps*ns,axis=1);a2[use]=np.sum(ps*ns**2,axis=1)
        if components:nodes,conditional=ns,ps
    norm=logsumexp(mass);w=np.exp(mass-norm)
    discarded_source=logsumexp(envelope[~keep]) if np.any(~keep) else -np.inf
    outside=np.maximum(0.,1-width/(hi-lo))
    tail=np.full_like(outside,-np.inf)
    np.log(outside,out=tail,where=outside>0)
    discarded_amount=logsumexp(envelope[ids]-amount_gap+tail)
    omitted=float(np.exp(np.logaddexp(discarded_source,discarded_amount)-norm))
    if omitted>limit:
        if np.isinf(source_gap): raise ValueError('Amount truncation bound exceeds limit')
        return bounded_posterior(L,K,order=order,support=support,source_gap=np.inf,
                                 amount_gap=amount_gap,limit=limit,components=components,analytic_moments=analytic_moments)
    return dict(ids=ids,w=w,nodes=nodes,conditional=conditional,
                a1=a1,a2=a2,omitted_bound=omitted,_fixed=None,_order=order,
                log_evidence=float(norm-np.log(len(L))),analytic_count=int(analytic.sum()))


class ClockDynamics:
    def __init__(self, model, velocity, *, tick=.0125, half_life=6., transport='zero_inflow_v1'):
        self.model=model;self.tick=tick;self.velocity=tuple(velocity)
        ref=DecayGaussianBelief(model,half_life_seconds=half_life,transport_id=transport)
        self.kernel=ref.kernel;self.P0=ref.covariance
        self.A,self.source_dt,count=transport_step(model,self.velocity,tick,transport,ref.decay_rate)
        if count!=1: raise ValueError('Clock exceeds model transition stability limit')
        _,_,noise_dt=decay_factors(ref.decay_rate,tick)
        self.Q=model.process_std_per_sqrt_second**2*noise_dt*self.kernel
        self.noise_factor=model.process_std_per_sqrt_second*math.sqrt(noise_dt)*np.linalg.cholesky(self.kernel)
        self.initial_factor=np.linalg.cholesky(self.P0)

    def tick_of(self,seconds):
        k=int(round(seconds/self.tick))
        if abs(k*self.tick-seconds)>1e-8: raise ValueError(f'Time is off fixed clock: {seconds}')
        return k

    def means(self,means,emission,steps=1):
        for _ in range(steps): means=(self.A@means.T).T+self.source_dt*emission
        return means


class ClockAffineFilter:
    """Many evidence histories sharing one covariance/response and source lattice."""
    def __init__(self,dynamics,emissions,histories=1):
        self.d=dynamics;self.E=emissions;n=len(dynamics.P0)
        self.m=np.full((histories,n),dynamics.model.initial_mean)
        self.P=dynamics.P0.copy();self.B=np.zeros((n,n));self.L=np.zeros((histories,len(emissions)))
        self.K=np.zeros(len(emissions));self.tick=0

    def advance(self,target):
        if target<self.tick: raise ValueError('Filter cannot advance backwards')
        for _ in range(target-self.tick):
            self.m=(self.d.A@self.m.T).T
            self.B=self.d.A@self.B
            self.B.flat[::len(self.B)+1]+=self.d.source_dt
            self.P=(self.d.A@(self.d.A@self.P).T).T+self.d.Q
        self.tick=target

    def condition(self,values,h):
        model=self.d.model;cross=self.P@h
        v=float(h@cross+(model.sensor_noise_std/model.gain)**2)
        residual=(np.asarray(values)-model.offset)/model.gain-self.m@h
        sensitivity=h@self.B;response=self.E@sensitivity
        self.L+=residual[:,None]*response[None]/v;self.K+=.5*response**2/v
        self.m+=residual[:,None]*(cross/v)[None]
        self.B-=np.outer(cross/v,sensitivity);self.P-=np.outer(cross,cross)/v
        self.P=(self.P+self.P.T)/2

    def posterior(self,history,fixed,order,components=False,analytic_moments=False):
        return bounded_posterior(self.L[history],self.K,fixed=fixed,order=order,components=components,analytic_moments=analytic_moments)

    def summary(self,history,p):
        emission=(p['w']*p['a1'])@self.E[p['ids']]
        return self.m[history]+self.B@emission,emission

    def validation(self,history,p,H,truth):
        if p['nodes'] is None:p=self.posterior(history,p['_fixed'],p['_order'],components=True)
        g=self.E[p['ids']]@(self.B.T@H.T)
        base=H@self.m[history]
        within=np.einsum('ij,jk,ik->i',H,self.P,H)
        mean=base+(p['w']*p['a1'])@g
        variance=np.maximum(0.,(p['w']*p['a2'])@(g*g)-(mean-base)**2)+within
        cdf=np.zeros(len(H))
        for start in range(0,len(g),256):
            sl=slice(start,start+256)
            z=(truth[None,None]-base[None,None]-g[sl,None]*p['nodes'][sl,:,None])/np.sqrt(within)[None,None]
            cdf+=np.einsum('i,ij,ijk->k',p['w'][sl],p['conditional'][sl],ndtr(z))
        return {'mean':mean,'variance':variance,'pit':cdf,'squared_error':(mean-truth)**2}


class OnlineAmountBelief:
    """Acquisition-interface adapter for unchanged scanner policy and mechanics."""
    def __init__(self,dynamics,sources,emissions,*,fixed=None,amount_order=128,prune=1e-12):
        self.config=dynamics.model;self.d=dynamics;self.sources=sources;self.E=emissions
        self.filter=ClockAffineFilter(dynamics,emissions)
        self.fixed=fixed;self.amount_order=amount_order;self.prune=prune
        self._p=None;self._summary=None;self._emission=None

    @property
    def time_seconds(self): return self.filter.tick*self.d.tick

    def clone(self):
        result=copy.copy(self);result.filter=copy.copy(self.filter)
        for key in ['m','P','B','L','K']: setattr(result.filter,key,getattr(self.filter,key).copy())
        return result

    def posterior(self):
        if self._p is None:self._p=self.filter.posterior(0,self.fixed,self.amount_order,analytic_moments=True)
        return self._p

    def advance(self,target,wind,source=None):
        if source is not None: raise ValueError('Truth source must not enter belief adapter')
        if not np.allclose(wind.velocity_yx,self.d.velocity,atol=1e-14,rtol=0) and target>self.time_seconds+1e-12:
            raise ValueError('Fixed-wind experiment received a different wind')
        self.filter.advance(self.d.tick_of(target));self._summary=None

    def assimilate(self,reading,wind,source=None):
        self.advance(reading.acquired_at_seconds,wind,source)
        h=observation_weights(self.config.grid_shape,reading.position_yx,reading.footprint)
        self.filter.condition([reading.value],h);self._p=None;self._summary=None;self._emission=None

    def summary(self):
        if self._summary is None:
            if self._emission is None:
                self._summary=self.filter.summary(0,self.posterior());self._emission=self._summary[1]
            else:self._summary=(self.filter.m[0]+self.filter.B@self._emission,self._emission)
        return self._summary

    def distribution_on(self,shape,*,fast_projection=False):
        p=self.filter.posterior(0,self.fixed,self.amount_order,components=True);joint=p['w'][:,None]*p['conditional']
        w=joint.ravel();order=np.argsort(w);cumulative=np.cumsum(w[order])
        keep=order[cumulative>self.prune]
        weights=w[keep];weights/=weights.sum()
        src,amount=np.unravel_index(keep,joint.shape)
        unique,inverse=np.unique(src,return_inverse=True)
        components=self.E[p['ids'][unique]]@self.filter.B.T
        ids,iw=interpolation_stencil(grid_positions(shape),self.config.grid_shape)
        projected=sum(components[:,ids[:,k]]*iw[:,k] for k in range(4))
        base=sum(self.filter.m[0,ids[:,k]]*iw[:,k] for k in range(4))
        variance=np.zeros(len(ids))
        for a in range(4):
            for b in range(4):variance+=iw[:,a]*iw[:,b]*self.filter.P[ids[:,a],ids[:,b]]
        amplitudes=p['nodes'][src,amount]
        if len(weights)>10000:
            return StreamingMarginals(self.time_seconds,base,projected,inverse,amplitudes,weights,variance,shape)
        mean=base+amplitudes[:,None]*projected[inverse]
        return PredictiveDistribution(self.time_seconds,weights,mean.reshape((-1,*shape)),
             np.broadcast_to(variance.reshape(shape),(len(weights),*shape)))

    def aggregate_moments_on(self,shape):
        # Used only for diagnostics; policy quantiles use the actual distribution.
        p=self.posterior();mean,_=self.summary()
        ids,iw=interpolation_stencil(grid_positions(shape),self.config.grid_shape)
        projected=sum(mean[ids[:,k]]*iw[:,k] for k in range(4))
        base=sum(self.filter.m[0,ids[:,k]]*iw[:,k] for k in range(4))
        within=np.zeros(len(ids))
        for a in range(4):
            for b in range(4):within+=iw[:,a]*iw[:,b]*self.filter.P[ids[:,a],ids[:,b]]
        if not np.any(self.filter.B):return projected.reshape(shape),within.reshape(shape)
        g=self.E[p['ids']]@self.filter.B.T
        g=sum(g[:,ids[:,k]]*iw[:,k] for k in range(4))
        variance=within+(p['w']*p['a2'])@(g*g)-(projected-base)**2
        return projected.reshape(shape),np.maximum(0.,variance).reshape(shape)

    moments_on=aggregate_moments_on

    def view(self):
        # Observer views do not calculate the expensive field-mixture variance at
        # every delivered reading; it is explicitly unavailable, not a zero band.
        mean,_=self.summary();shape=self.config.grid_shape
        positive=np.maximum(mean,0);mass=positive/positive.sum() if positive.sum()>0 else np.zeros_like(positive)
        return BeliefView(self.time_seconds,mean.reshape(shape).copy(),None,mass.reshape(shape))
