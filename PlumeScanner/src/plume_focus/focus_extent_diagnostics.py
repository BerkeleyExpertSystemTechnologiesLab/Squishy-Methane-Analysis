"""Evaluator-only fixed-evidence probes. Never imported by runtime planning."""
from dataclasses import replace
from statistics import NormalDist
import numpy as np
from scipy.ndimage import label

from plume_focus.acquisition import AcquisitionController
from plume_focus.full_fov_loop import FullFovPlan
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt, SensorReading, Footprint
from plume_focus.signal_belief import GaussianSignalBelief, grid_positions, observation_weights

THRESHOLD = .15
Z95 = NormalDist().inv_cdf(.95)


class EvidenceReplay(AcquisitionController):
    """Observer adapter following recorded paths, independent of allocation.

    Uses shared acquisition validation/replay. Alternative source assumptions do
    not generate plans. This adapter is not a production controller entry point.
    """
    def __init__(self, session, *, source_off=False):
        self._initialize_acquisition(session.instrument.capabilities(), session.model,
                                     None if source_off else session.source_position,
                                     session.instrument)
        self._belief = GaussianSignalBelief(session.model, transport_id=session.transport_id)

    def decision(self, event):
        if self._active is not None:
            raise ValueError("Recorded decision overlaps an open scan")
        self._belief.advance(event['at_seconds'], self._wind, self.source_position)
        self._wind = CausalWind.from_dict(event['wind'])
        self._last_available = event['at_seconds']

    def event(self, event):
        if event['operation'] == 'plan_next':
            self.decision(event)
            from plume_focus.full_fov_execution import decode_plan
            plan = decode_plan(event['plan'])
            if plan.samples:
                self._open_scan(plan)
        elif event['operation'] == 'ingest_readings':
            self.ingest_readings([SensorReading.from_dict(r) for r in event['readings']],
                                 available_at_seconds=event['at_seconds'])
        elif event['operation'] == 'finish_scan':
            self._finish_evidence(ExecutionReceipt.from_dict(event['receipt']))
            self._close_scan()
        else:
            raise ValueError("Unknown recorded operation")


def select_events(record):
    """Freeze prescribed events without consulting diagnostic outcomes."""
    decisions = [(i, e) for i, e in enumerate(record['events']) if e['operation'] == 'plan_next']
    first_full = next((s for s in record['scans'] if s['startup_complete']), None)
    targets = [('post_startup', first_full['receipt']['received_at_seconds'] if first_full else None),
               ('after_180', 180.)]
    if record['job']['case'] == 'cessation':
        scan = next((s for s in record['scans'] if s['plan']['mode'] == 'FULL'
                     and s['plan']['started_at_seconds'] >= 180 and not s['missing_full_cells']), None)
        targets.append(('post_cessation_rescan', scan['receipt']['received_at_seconds'] if scan else None))
    found, unavailable = {}, []
    for name, at in targets:
        hit = next(((i, e) for i, e in decisions if at is not None and e['at_seconds'] >= at-1e-9), None)
        if hit is None:
            unavailable.append(name)
        else:
            found.setdefault(hit[0], []).append(name)
    return [{'event_index': i, 'roles': roles, 'at_seconds': record['events'][i]['at_seconds']}
            for i, roles in sorted(found.items())], unavailable


def region(mask, values, wind):
    mask = np.asarray(mask, bool)
    pos = grid_positions(mask.shape)
    direction = np.asarray(wind, float)
    direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) else np.array([0., 1.])
    normal = np.array([-direction[1], direction[0]])
    def extent(m):
        points = pos[m.ravel()]
        if not len(points):
            return {'area': 0., 'downwind_extent': None, 'crosswind_extent': None}
        # Include cell dimensions, not just the distance between cell centers.
        pitch = 1 / np.array(mask.shape)
        return {'area': float(m.mean()),
                'downwind_extent': float(np.ptp(points @ direction) + np.abs(direction) @ pitch),
                'crosswind_extent': float(np.ptp(points @ normal) + np.abs(normal) @ pitch)}
    components, count = label(mask)
    peak_id = int(components.ravel()[np.argmax(values)])
    return {**extent(mask), 'cells': int(mask.sum()), 'components': int(count),
            'peak_component': extent(components == peak_id) if peak_id else extent(np.zeros_like(mask))}


def surfaces(belief, shape, wind, truth=None):
    mean, variance = belief.moments_on(shape)
    upper = mean + Z95 * np.sqrt(variance)
    mm, um = mean >= THRESHOLD, upper >= THRESHOLD
    extra = um & ~mm
    result = {'peak_mean': float(mean.max()), 'peak_upper': float(upper.max()),
              'mean': region(mm, mean, wind), 'upper': region(um, upper, wind),
              'bright_core': region(mean >= .5*mean.max(), mean, wind) if mean.max() > 0 else None,
              'uncertainty_only_cells': int(extra.sum()), 'average_variance': float(variance.mean())}
    if truth is not None:
        signal = truth >= THRESHOLD
        result.update(truth=region(signal, truth, wind), truth_peak=float(truth.max()),
            mean_squared_error=float(np.mean((mean-truth)**2)),
            mean_background_cells=int((mm & ~signal).sum()),
            uncertainty_added_background_cells=int((extra & ~signal).sum()),
            uncertainty_added_signal_cells=int((extra & signal).sum()),
            missed_signal_cells=int((signal & ~um).sum()),
            upper_exceedance_fraction=float(np.mean(truth > upper)),
            captured_signal_sum=float(truth[um].sum()), total_signal_sum=float(truth.sum()))
    return result, mean, upper


def forecast(belief, horizon, wind, source, variant='baseline'):
    clone = belief.clone()
    if variant in ('diffusion_off', 'both_off'):
        clone.config = replace(clone.config, diffusivity=0.)
    if variant in ('process_off', 'both_off'):
        clone.config = replace(clone.config, process_std_per_sqrt_second=0.)
    clone.advance(clone.time_seconds+horizon, wind, None if variant == 'future_source_off' else source)
    return clone


def controlled_readings(config, footprint):
    records = []
    for variant in ('baseline', 'half_correlation', 'double_grid', 'double_footprint'):
        model = config
        fp = footprint
        if variant == 'half_correlation': model = replace(config, correlation_length=config.correlation_length/2)
        if variant == 'double_grid': model = replace(config, grid_shape=tuple(2*n for n in config.grid_shape))
        if variant == 'double_footprint': fp = Footprint(min(1.,2*fp.height), min(1.,2*fp.width))
        for value in (0., .1, .3, 1.):
            b = GaussianSignalBelief(model, transport_id='zero_inflow_v1')
            h = observation_weights(model.grid_shape, (.5,.25), fp)
            p, mu = b.covariance.copy(), b.mean.copy()
            cross = p @ h
            expected = mu + cross*(value-h@mu)/(h@cross+model.sensor_noise_std**2)
            b.condition(value,h,model.sensor_noise_std**2)
            stats, _, _ = surfaces(b,(120,160),(0.,1.))
            records.append({'variant':variant,'reading':value,'analytic_max_error':float(np.max(abs(b.mean-expected))),
                            **stats})
    return records
