"""Independent arithmetic, forecast and numerical-confounder observers."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import write_json
from plume_focus.signal_belief import GaussianSignalBelief, SignalModelConfig, grid_positions, observation_weights, footprint_points
from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.sensing_counterfactual import TruthQuadrature, predicted_mean, run_counterfactual
from plume_focus.sensing_risk import projection_gram, score_pattern


def runner():
    spec=importlib.util.spec_from_file_location("diagnosis_runner",Path(__file__).with_name("run-scan-value-diagnosis.py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def arithmetic():
    # Independent joint Gaussian conditioning, including correlated neighbors.
    p=np.array([[.2,.1],[.1,.3]]);mu=np.array([.1,.2]);h=np.array([[1.,0.],[.25,.75],[1.,0.]])
    noise=.03;y=np.array([.8,0.,.7])
    system=h@p@h.T+noise*np.eye(3)
    batch_mu=mu+p@h.T@np.linalg.solve(system,y-h@mu)
    batch_p=p-p@h.T@np.linalg.solve(system,h@p)
    seq_mu,seq_p=mu.copy(),p.copy()
    for obs,value in zip(h,y):
        cross=seq_p@obs;var=obs@cross+noise
        seq_mu+=cross*(value-obs@seq_mu)/var;seq_p-=np.outer(cross,cross)/var
    np.testing.assert_allclose(seq_mu,batch_mu,atol=1e-13)
    np.testing.assert_allclose(seq_p,batch_p,atol=1e-13)
    rng=np.random.default_rng(721)
    x=rng.multivariate_normal(mu,p,20000);obs=x@h.T+rng.normal(0,np.sqrt(noise),(len(x),3))
    estimates=mu+(obs-h@mu)@np.linalg.solve(system,h@p)
    observed=float(np.mean((x-estimates)**2));expected=float(np.trace(batch_p)/2)
    return {"batch_mean":batch_mu.tolist(),"batch_covariance":batch_p.tolist(),
            "expected_posterior_mse":expected,"model_consistent_sampled_mse":observed,
            "relative_difference":abs(observed-expected)/expected,"samples":len(x)}


def boundary():
    model=SignalModelConfig(grid_shape=(12,16),diffusivity=0,process_std_per_sqrt_second=0,source_rate=0)
    rows=[]
    for transport in ("legacy_clipped_v1","zero_inflow_v1"):
        for axis in (0,1):
            for sign in (-1,1):
                state=GaussianSignalBelief(model,transport_id=transport);state.mean[:]=0
                grid=state.mean.reshape(model.grid_shape)
                if axis==0:grid[0 if sign>0 else -1,:]=1
                else:grid[:,0 if sign>0 else -1]=1
                wind=[0.,0.];wind[axis]=sign*.04;initial=float(state.mean.mean())
                for horizon in (0.,6.,12.,30.):
                    future=state.clone();future.advance(horizon,CausalWind(velocity_yx=tuple(wind)))
                    rows.append({"transport":transport,"axis":axis,"sign":sign,"horizon":horizon,
                        "initial_integral":initial,"integral":float(future.mean.mean()),
                        "minimum":float(future.mean.min()),"maximum":float(future.mean.max())})
    return rows


def forecast_controls(root):
    r=runner();declaration=r.read(Path(root)/"declaration.json");rows=[];exposure=[]
    for name,ref in declaration["fixtures"].items():
        if not ref["available"]:continue
        for transport in ("legacy_clipped_v1","zero_inflow_v1"):
            directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
            controller=snapshot.fork_controller(SensingController);start=controller._last_available
            base=controller._belief;gram=projection_gram(config.model.grid_shape,(120,160))
            shape=(120,160);points=grid_positions(shape)
            for horizon in (0.,1.,3.,6.,12.):
                future=base.clone();future.advance(start+horizon,controller._wind,controller.source_position)
                partitioned=base.clone()
                if horizon:
                    for t in np.linspace(start,start+horizon,int(round(horizon/.05))+1)[1:]:
                        partitioned.advance(float(t),controller._wind,controller.source_position)
                mu,var=future.moments_on(shape)
                true=config.source.build().signal_at(np.full(len(points),start+horizon),points).reshape(shape)
                error=mu-true;dy,dx=np.gradient(true);gradient=np.sqrt(dx*dx+dy*dy)
                signal=true>config.session.allocation_threshold
                physical=controller.visit_times
                iy=np.minimum((points[:,0]*physical.shape[0]).astype(int),physical.shape[0]-1)
                ix=np.minimum((points[:,1]*physical.shape[1]).astype(int),physical.shape[1]-1)
                visits=physical[iy,ix].reshape(shape);age=np.where(visits>=0,start+horizon-visits,np.inf)
                masks={"signal":signal,"background":~signal,"high_gradient":gradient>np.quantile(gradient,.9),
                       "older_30s":age>30,"recent_10s":age<=10}
                groups={}
                for key,mask in masks.items():
                    groups[key]={"fraction":float(mask.mean()),"mse":float(np.mean(error[mask]**2)) if mask.any() else None,
                        "predicted_variance":float(np.mean(var[mask])) if mask.any() else None}
                delta=partitioned.mean-future.mean
                risk=float(np.sum(gram*future.covariance));split_risk=float(np.sum(gram*partitioned.covariance))
                rows.append({"fixture":name,"transport":transport,"horizon":horizon,
                    "actual_mse":float(np.mean(error**2)),"predicted_variance":float(np.mean(var)),"groups":groups,
                    "partition_mean_mse":float(delta@gram@delta),"partition_risk_difference":split_risk-risk,
                    "partition_risk_relative_difference":abs(split_risk-risk)/max(1e-10,abs(risk)),
                    "motion":future.diagnostics() if hasattr(future,"diagnostics") else None})
            # Truth belongs only to this observer. Noise disabled solely to measure
            # exposure operator discrepancy, not to change the execution study.
            scanner=FakeScanner(config.source.build(),noise_std=0,gain=config.model.gain,offset=config.model.offset)
            for candidate in candidates:
                execution=scanner.execute(candidate.plan)
                residual=[]
                for reading in execution.readings:
                    p=footprint_points(reading.position_yx,reading.footprint)
                    value=np.mean(config.source.build().signal_at(np.full(len(p),reading.acquired_at_seconds),p))
                    residual.append((reading.value-config.model.offset)/config.model.gain-value)
                exposure.append({"fixture":name,"transport":transport,"candidate":candidate.name,
                    "rms_exposure_minus_midpoint":float(np.sqrt(np.mean(np.square(residual)))),
                    "max_absolute":float(np.max(np.abs(residual))),"assumed_noise_std":config.model.sensor_noise_std})
        print("controls",name,flush=True)
    return rows,exposure


def verify(root):
    path=Path(root)/"controls.json"
    r=runner();forecasts,exposure=forecast_controls(root)
    result={"schema_version":1,"producing_revision":r.revision(),"arithmetic":arithmetic(),
            "boundary":boundary(),"forecasts":forecasts,"exposure":exposure}
    write_json(path,result)
    return result


def refine_fixture(root,name,transport):
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    output=directory/f"{transport}-refinement.json"
    if output.exists():return r.read(output)
    controller=snapshot.fork_controller(SensingController);base=r.score_fixture(root,name,transport)
    risks=np.array([x['risk'] for x in base['candidates']])
    # All alternatives on one fixture per physical profile, plus blank, rather
    # than an outcome-selected favorable subset of paths.
    levels=[];prior=risks;accepted=None
    for step in (.05,.025):
        session=replace(config.session,policy=replace(config.session.policy,risk_step_seconds=step))
        current=[]
        for candidate in candidates:
            current.append(score_pattern(controller._belief,session,controller._wind,
                r.pattern(controller,config,candidate,tuple(payload['required'])),12.,seed=0)['risk'])
        current=np.asarray(current)
        tolerance=np.maximum(1e-10,.01*np.abs(current))
        regret=float(current[int(np.argmin(prior))]-current.min())
        passed=bool(np.all(np.abs(prior-current)<=tolerance) and regret<=max(1e-10,.01*abs(float(current.min()))))
        levels.append({"step":step,"risks":current.tolist(),"maximum_absolute_change":float(np.max(np.abs(prior-current))),
                       "selected_regret":regret,"passed":passed})
        if passed:accepted=.1 if step==.05 else .05;break
        prior=current
    partition_session=replace(config.session,model=replace(config.model,max_transition_seconds=.05))
    partition_risks=[]
    for candidate in candidates:
        partition_risks.append(score_pattern(controller._belief,partition_session,controller._wind,
            r.pattern(controller,config,candidate,tuple(payload['required'])),12.,seed=0)['risk'])
    partition_risks=np.asarray(partition_risks)
    partition_regret=float(partition_risks[int(np.argmin(risks))]-partition_risks.min())
    result={"producing_revision":r.revision(),"accepted_risk_step":accepted,"checks":levels,
            "partition_check":{"maximum_step_seconds":.05,"risks":partition_risks.tolist(),
                "selected_candidate_regret":partition_regret,
                "material_rank_change":partition_regret>max(1e-10,.01*abs(float(partition_risks.min()))),
                "maximum_relative_risk_change":float(np.max(np.abs(partition_risks-risks)/np.maximum(1e-10,np.abs(risks))))},
            "status":"converged" if accepted else "numerically_unresolved"}
    write_json(output,result);return result


def refine_mixture(root,name,transport):
    """Retain all 49 components; refine only scoring integration and histories."""
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    output=directory/f'{transport}-mixture-refinement.json'
    if output.exists():return r.read(output)
    controller=snapshot.fork_controller(SensingController);checks=[]
    def scores(step,histories):
        session=replace(config.session,policy=replace(config.session.policy,
                        risk_step_seconds=step,history_count=histories))
        return np.array([score_pattern(controller._belief,session,controller._wind,
            r.pattern(controller,config,c,tuple(payload['required'])),12.,seed=0)['risk'] for c in candidates])
    def compare(previous,current,**settings):
        regret=float(current[int(np.argmin(previous))]-current.min())
        passed=bool(np.all(np.abs(previous-current)<=np.maximum(1e-10,.01*np.abs(current)))
                    and regret<=max(1e-10,.01*abs(float(current.min()))))
        checks.append({**settings,'risks':current.tolist(),'passed':passed,'selected_regret':regret,
                       'maximum_absolute_change':float(np.max(np.abs(previous-current)))})
        return passed
    base=scores(.1,32);prior=base;accepted_histories=None
    for histories in (64,128):
        current=scores(.1,histories)
        if compare(prior,current,kind='histories',histories=histories,step=.1):
            accepted_histories=histories//2;break
        prior=current
    histories=accepted_histories or 128
    prior=base if histories==32 else scores(.1,histories);accepted_step=None
    for step in (.05,.025):
        current=scores(step,histories)
        if compare(prior,current,kind='time',histories=histories,step=step):
            accepted_step=step*2;break
        prior=current
    result={'status':'converged' if accepted_histories and accepted_step else 'numerically_unresolved',
            'accepted_histories':accepted_histories,'accepted_step':accepted_step,'checks':checks,
            'snapshot_hash':snapshot.digest,'producing_revision':r.revision(),
            'wind_quadrature':'unresolved: retained 49 hypotheses; this check does not certify the bias integral'}
    write_json(output,result);return result


def compact_exposure_controls():
    """Independent time integral of a stationary/moving compact smooth signal."""
    rows=[];duration=.025;sigma=.01
    # A separable compact bump is evaluated at one spatial point. Its stationary
    # integral is exactly its midpoint; motion produces an exposure discrepancy.
    def bump(z):
        return np.where(np.abs(z)<1,np.exp(-1/np.maximum(1e-300,1-z*z)),0.)
    for speed in (0.,.04,.2):
        for offset in (0.,.5*sigma):
            values=[]
            for n in (1001,2001):
                times=np.linspace(-duration/2,duration/2,n)
                values.append(float(np.trapezoid(bump((offset-speed*times)/sigma),times)/duration))
            midpoint=float(bump(np.array(offset/sigma)))
            rows.append({'speed_fov_per_second':speed,'offset_fov':offset,
                'exposure':values[-1],'midpoint':midpoint,'difference':values[-1]-midpoint,
                'integral_refinement_error':abs(values[-1]-values[-2])})
    assert all(abs(row['difference'])<1e-12 for row in rows if row['speed_fov_per_second']==0)
    return rows


if __name__=="__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--phase',choices=('verify','refine','mixture-refine','compact-exposure'),default='verify')
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.phase=='verify':verify(args.output)
    elif args.phase=='compact-exposure':write_json(args.output/'compact-exposure.json',compact_exposure_controls())
    else:
        names=(['double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion-risk',
                'diagnostic-source-free-biased-scale-1-seed-0-motion-risk'] if args.phase=='mixture-refine' else
               ['coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk',
               'double-prior-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk',
               'diagnostic-source-free-biased-scale-0-seed-0-baseline-risk'])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(refine_mixture if args.phase=='mixture-refine' else refine_fixture,str(args.output),name,transport) for name in names
                     for transport in ('legacy_clipped_v1','zero_inflow_v1')]
            for future in as_completed(futures):print(future.result(),flush=True)
