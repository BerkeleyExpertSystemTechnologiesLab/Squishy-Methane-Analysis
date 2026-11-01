"""Local fixed-history amount controls and explicit source/amount quadrature."""
import hashlib, json
import numpy as np
from scipy.special import logsumexp, ndtr
from numpy.polynomial.legendre import leggauss
from plume_focus.model_consistent_control import MatchedFieldGenerator
from plume_focus.source_prior_belief import SourcePriorBank
from plume_focus.source_input import gaussian_source_input
from plume_focus.sensing_contracts import CausalWind, SensorReading


class InputMatchedGenerator(MatchedFieldGenerator):
    """Existing statistical generator with an explicitly declared source vector."""
    def __init__(self, model, source, *, total_rate, input_id, **kwargs):
        super().__init__(model, source, **kwargs)
        self.emission = gaussian_source_input(model.grid_shape, source,
            source_sigma=model.source_sigma, total_rate=total_rate, formulation_id=input_id)


class SourceAmountBank(SourcePriorBank):
    """Direct small-bank reference; preserve location/amount dependence."""
    def __init__(self, model, sources, multipliers, priors, *, total_rate, input_id, **kwargs):
        super().__init__(model, sources, priors, **kwargs)
        self.multipliers = np.asarray(multipliers, float)
        if self.multipliers.shape != (len(self.sources),):
            raise ValueError('One amount multiplier is required per source hypothesis')
        self.emissions = gaussian_source_input(model.grid_shape, self.sources,
            source_sigma=model.source_sigma, total_rate=total_rate*self.multipliers,
            formulation_id=input_id)


def amount_quadrature(order, lower, upper):
    if type(order) is not int or order < 2 or not 0 <= lower < upper:
        raise ValueError('Invalid amount quadrature')
    x, w = leggauss(order)
    return (lower+upper)/2+x*(upper-lower)/2, w/2


def generate_history(record, run, cfg, multiplier, seed):
    gen = InputMatchedGenerator(run.model, cfg['source_yx'], total_rate=cfg['reference_total_per_second']*multiplier,
        input_id=cfg['source_input_id'], half_life_seconds=run.session.half_life_seconds,
        transport_id=run.session.transport_id, seed=seed)
    wind=CausalWind(); events=[]; states=[]; signatures=[]
    for old in record['events']:
        op=old['operation'];at=old['at_seconds'];e={'operation':op,'at_seconds':at}
        if op=='ingest_readings':
            if len(old['readings']) != 1: raise ValueError('Singleton streaming history required')
            r=SensorReading.from_dict(old['readings'][0])
            gen.advance(r.acquired_at_seconds,wind)
            e['reading']={**old['readings'][0],'value':gen.observe(r)}
            gen.advance(at,wind)
        elif op=='plan_next':
            gen.advance(at,wind);wind=CausalWind.from_dict(old['wind']);e['wind']=old['wind']
        elif op=='finish_scan':
            gen.advance(at,wind);e['scan_id']=old['receipt']['scan_id']
        else: raise ValueError('Unknown schedule event')
        events.append(e);states.append(gen.state.copy())
        if gen.rng is not None:
            signatures.append(hashlib.sha256(json.dumps(gen.rng.bit_generator.state,sort_keys=True).encode()).hexdigest())
    for h in [0.]+cfg['forecast_horizons_seconds']:
        at=run.duration_seconds+h;gen.advance(at,wind)
        events.append({'operation':'end' if h==0 else 'forecast','at_seconds':at,'horizon_seconds':h})
        states.append(gen.state.copy())
        if gen.rng is not None:
            signatures.append(hashlib.sha256(json.dumps(gen.rng.bit_generator.state,sort_keys=True).encode()).hexdigest())
    return events,np.asarray(states),signatures


def source_amount_posterior(linear, quadratic, nodes, node_weights, *, fixed_amount=None,
                            gap=50., omitted_limit=1e-10, chunk=1024, amount_support=None):
    """Uniform location prior, likelihood exp(a L-a^2 K), optional amount point.

    Return source marginals and source-weighted first/second amount moments.
    Bounded screening cannot use evaluator truth. Its bound includes all omitted
    locations and the normalized amount prior.
    """
    linear=np.asarray(linear,float);quadratic=np.asarray(quadratic,float)
    if (linear.ndim!=1 or quadratic.shape!=linear.shape or len(linear)==0
            or not np.all(np.isfinite(linear)) or not np.all(np.isfinite(quadratic))
            or np.any(quadratic < -1e-10)):
        raise ValueError('Invalid likelihood statistics')
    quadratic=np.maximum(quadratic,0.)
    if fixed_amount is not None:
        if not np.isfinite(fixed_amount) or fixed_amount<0: raise ValueError('Invalid amount')
        logs=fixed_amount*linear-fixed_amount**2*quadratic
        w=np.exp(logs-logsumexp(logs))
        return {'w0':w,'w1':w*fixed_amount,'w2':w*fixed_amount**2,
            'amount_mean':float(fixed_amount),'amount_variance':0.,'amount_mass':None,
            'amount_interval90':[float(fixed_amount)]*2,'omitted_mass_bound':0.,'retained_locations':len(w)}
    nodes=np.asarray(nodes,float);node_weights=np.asarray(node_weights,float)
    if (nodes.ndim!=1 or node_weights.shape!=nodes.shape or len(nodes)<2
            or not np.all(np.isfinite(nodes)) or not np.all(nodes>=0)
            or not np.all(node_weights>0) or not np.isclose(node_weights.sum(),1.)):
        raise ValueError('Invalid amount prior quadrature')
    lower,upper=amount_support if amount_support is not None else (nodes.min(),nodes.max())
    if not 0<=lower<upper or nodes.min()<lower or nodes.max()>upper or gap<0 or omitted_limit<0:
        raise ValueError('Invalid support or screening limit')
    # Maximize over the complete declared support, including endpoints.
    optimum=np.divide(linear,2*quadratic,out=np.zeros_like(linear),where=quadratic>0)
    optimum=np.clip(optimum,lower,upper)
    optimum=np.where(quadratic==0,np.where(linear>=0,upper,lower),optimum)
    envelope=optimum*linear-optimum**2*quadratic
    if np.any(~np.isfinite(envelope)):
        keep=np.ones(len(linear),bool)
    else: keep=envelope>=envelope.max()-gap
    ids=np.flatnonzero(keep);logprior=np.log(node_weights)
    logm=np.full(len(linear),-np.inf);conditional1=np.zeros(len(linear));conditional2=np.zeros(len(linear))
    for begin in range(0,len(ids),chunk):
        ii=ids[begin:begin+chunk]
        a=linear[ii,None]*nodes-quadratic[ii,None]*nodes**2+logprior
        z=logsumexp(a,axis=1);p=np.exp(a-z[:,None])
        logm[ii]=z;conditional1[ii]=p@nodes;conditional2[ii]=p@(nodes**2)
    normalizer=logsumexp(logm)
    omitted_upper=logsumexp(envelope[~keep]) if np.any(~keep) else -np.inf
    bound=float(np.exp(omitted_upper-np.logaddexp(omitted_upper,normalizer))) if np.isfinite(omitted_upper) else 0.
    if bound>omitted_limit:
        return source_amount_posterior(linear,quadratic,nodes,node_weights,gap=np.inf,
                                       omitted_limit=omitted_limit,chunk=chunk,amount_support=(lower,upper))
    w=np.exp(logm-normalizer);amount_mass=np.zeros(len(nodes))
    for begin in range(0,len(ids),chunk):
        ii=ids[begin:begin+chunk]
        a=linear[ii,None]*nodes-quadratic[ii,None]*nodes**2+logprior
        amount_mass+=np.exp(a-logm[ii,None]).T@w[ii]
    mean=float(w@conditional1);second=float(w@conditional2)
    cdf=np.cumsum(amount_mass)
    interval=[float(nodes[min(np.searchsorted(cdf,p),len(nodes)-1)]) for p in [.05,.95]]
    return {'w0':w,'w1':w*conditional1,'w2':w*conditional2,'amount_mean':mean,
        'amount_variance':max(0.,second-mean**2),'amount_mass':amount_mass,
        'amount_interval90':interval,'omitted_mass_bound':bound,'retained_locations':len(ids)}


def forecast_footprint_cdf(posterior, linear, quadratic, nodes, node_weights,
                           response, base, within_std, truth, fixed_amount=None):
    if within_std<=0: raise ValueError('Positive conditional standard deviation required')
    w=posterior['w0']
    if fixed_amount is not None:
        return float(w@ndtr((truth-base-fixed_amount*response)/within_std))
    ids=np.flatnonzero(w);value=0.;logprior=np.log(node_weights)
    for begin in range(0,len(ids),1024):
        ii=ids[begin:begin+1024]
        a=linear[ii,None]*nodes-quadratic[ii,None]*nodes**2+logprior
        conditional=np.exp(a-logsumexp(a,axis=1)[:,None])
        cdf=ndtr((truth-base-response[ii,None]*nodes)/within_std)
        value+=float(w[ii]@np.sum(conditional*cdf,axis=1))
    return value
