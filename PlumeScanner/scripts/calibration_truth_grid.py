"""Exact separable evaluation of the existing synthetic source on regular grids.

Only evaluator quadrature uses this path. Sensor readings still use signal_at.
Scalar puff parameters avoid recomputing the same trigonometry per mesh pixel.
"""
import numpy as np


def clean_grid(config,time,shape):
    c=config.source
    if c.signal_scale==0 or (config.cessation_seconds is not None and time>=config.cessation_seconds):
        return np.zeros(shape)
    if config.active_windows is not None and not any(a<=time<b for a,b in config.active_windows):
        return np.zeros(shape)
    t=time*c.fps
    y=(np.arange(shape[0])+.5)/shape[0]*(c.shape[0]-1)
    x=(np.arange(shape[1])+.5)/shape[1]*(c.shape[1]-1)
    wind=np.array(c.wind_velocity);direction=wind/max(np.linalg.norm(wind),1e-9)
    normal=np.array([-direction[1],direction[0]])
    first=-c.memory_frames+max(0,int(np.ceil(t/c.emission_interval)))*c.emission_interval
    result=np.zeros(shape)
    for birth in range(first,int(np.ceil(t))+1,c.emission_interval):
        age=t-birth
        if not 0<=age<=c.memory_frames:continue
        lateral=2.8*np.sin(.31*age+.47*birth+c.lateral_phase)+1.2*np.sin(.11*t+.5*c.lateral_phase)
        center=np.array(c.source)+wind*age+normal*lateral
        sy=c.base_sigma+c.diffusion_per_frame*age
        sx=c.base_sigma*1.35+c.diffusion_per_frame*age*1.65
        amp=210*np.exp(-age/52)*(.9+.1*np.sin(.37*birth+c.amplitude_phase))
        result+=amp*np.exp(-((y-center[0])**2/(2*sy**2))[:,None]-((x-center[1])**2/(2*sx**2))[None,:])
    shimmer=1+.06*np.sin(.17*x[None,:]+.19*t+c.shimmer_phase)+.04*np.cos(.13*y[:,None]+.5*c.shimmer_phase)
    return result*shimmer/255*c.signal_scale
