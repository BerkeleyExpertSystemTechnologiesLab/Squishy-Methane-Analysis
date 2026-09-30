"""Causal decay-aware loss and affine fitting; evaluator only."""
from functools import lru_cache
import math
import numpy as np
from plume_focus.calibrated_scanner import DecayGaussianBelief, decay_factors
from plume_focus.sensing_risk import linear_transition
from plume_focus.focus_extent_diagnostics import EvidenceReplay


@lru_cache(maxsize=256)
def mean_steps(config,velocity,source,elapsed,transport_id,half_life):
    speed=np.max(np.abs(velocity)*np.array(config.grid_shape))
    diffusion=config.diffusivity*sum(n*n for n in config.grid_shape)
    max_dt=min(config.max_transition_seconds,.5/max(speed,1e-12),.45/max(diffusion,1e-12))
    count=max(1,int(math.ceil(elapsed/max_dt)));dt=elapsed/count
    operator,emission,_=linear_transition(config,velocity,source,dt,transport_id)
    rate=0. if half_life is None else math.log(2)/half_life
    a,source_dt,_=decay_factors(rate,dt)
    return operator*a,emission*(source_dt/dt),count


def mean_forecast(belief,mean,target,wind,source):
    elapsed=max(0.,target-belief.time_seconds)
    if target<belief.time_seconds-1e-10:raise ValueError('Cannot forecast backwards')
    if elapsed<=1e-12:return mean.copy()
    operator,emission,count=mean_steps(belief.config,tuple(wind.velocity_yx),source,round(elapsed,12),
                                     belief.transport_id,belief.half_life_seconds)
    result=mean.copy()
    for _ in range(count):result=operator@result+emission
    return result


def predicted_mean(controller,target):
    if target<controller._last_available-1e-9:raise ValueError('Past online estimate requires replay')
    b=controller._belief
    return mean_forecast(b,b.mean,target,controller._wind,controller.source_position)


class AffineBelief(DecayGaussianBelief):
    """Rate zero and upper-endpoint means share one value-independent covariance."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.mean_zero=self.mean.copy()

    def clone(self):
        b=super().clone();b.mean_zero=self.mean_zero.copy();return b

    def advance(self,target,wind,source=None):
        zero=mean_forecast(self,self.mean_zero,target,wind,None)
        super().advance(target,wind,source);self.mean_zero=zero

    def condition(self,value,weights,noise_variance):
        h=np.asarray(weights);cross=self.covariance@h
        zero=self.mean_zero+cross*((value-h@self.mean_zero)/(h@cross+noise_variance))
        super().condition(value,weights,noise_variance);self.mean_zero=zero


class AffineReplay(EvidenceReplay):
    def __init__(self,session):
        super().__init__(session)
        self._belief=AffineBelief(session.model,half_life_seconds=session.half_life_seconds,
                                 transport_id=session.transport_id)


class AffineLoss:
    def __init__(self,truth):self.truth=truth;self.index=0;self.coefficients=np.zeros(3)

    def before(self,c,at):
        q=self.truth;b=c._belief
        while self.index<len(q.times) and q.times[self.index]<at-1e-10:
            i=self.index;t=float(q.times[i]);self.index+=1
            mu0=mean_forecast(b,b.mean_zero,t,c._wind,None)
            delta=mean_forecast(b,b.mean,t,c._wind,c.source_position)-mu0
            self.coefficients+=q.durations[i]*np.array([
                delta@q.gram@delta, delta@(q.gram@mu0-q.cross[i]),
                mu0@q.gram@mu0-2*mu0@q.cross[i]+q.square[i]])


def fit_rate(coefficients,upper=.30):
    a,b,c=np.asarray(coefficients,float)
    fraction=float(np.clip(-b/a,0,1)) if a>1e-20 else 0.
    return {'source_rate':fraction*upper,'loss':float(a*fraction*fraction+2*b*fraction+c),
            'coefficients':[float(a),float(b),float(c)],'fraction':fraction}
