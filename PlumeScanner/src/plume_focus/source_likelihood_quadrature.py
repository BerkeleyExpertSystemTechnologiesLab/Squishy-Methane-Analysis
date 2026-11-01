"""Exact affine source-input likelihood for offline quadrature refinement.

For fixed dynamics/H/R, conditional mean = m + B e(source). Innovations yield
log L(e) = c + e.z - .5 e.A.e. No posterior/moment approximation is made.
"""
import math
import numpy as np
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.source_prior_belief import transport_step


class AffineSourceLikelihood:
    def __init__(self, model, *, half_life_seconds, transport_id):
        self.reference=DecayGaussianBelief(model,half_life_seconds=half_life_seconds,transport_id=transport_id)
        n=len(self.reference.mean)
        self.response=np.zeros((n,n));self.precision=np.zeros((n,n));self.linear=np.zeros(n);self.constant=0.

    def advance(self,target,wind):
        r=self.reference;elapsed=target-r.time_seconds
        if elapsed < -1e-10:raise ValueError('Cannot go backwards')
        if elapsed>1e-12:
            operator,source_dt,count=transport_step(r.config,tuple(wind.velocity_yx),elapsed,r.transport_id,r.decay_rate)
            for _ in range(count):self.response=operator@self.response+source_dt*np.eye(len(r.mean))
        r.advance(target,wind,None)

    def condition(self,value,h,noise_variance):
        r=self.reference;cross=r.covariance@h;variance=float(h@cross+noise_variance)
        residual=value-float(h@r.mean);sensitivity=h@self.response
        self.precision+=np.outer(sensitivity,sensitivity)/variance
        self.linear+=sensitivity*residual/variance
        self.constant+=-.5*(math.log(2*math.pi*variance)+residual**2/variance)
        self.response-=np.outer(cross/variance,sensitivity)
        r.condition(value,h,noise_variance)

    def log_likelihood(self,emissions):
        return self.constant+emissions@self.linear-.5*np.sum((emissions@self.precision)*emissions,axis=1)

    def means(self,emissions):
        return self.reference.mean[None]+emissions@self.response.T
