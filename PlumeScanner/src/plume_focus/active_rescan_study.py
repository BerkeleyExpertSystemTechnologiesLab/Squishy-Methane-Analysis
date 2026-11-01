"""Bounded paired toy study. Simulator/evaluator truth is confined here."""
import argparse
from copy import deepcopy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from plume_focus.active_rescan import (VisibleState, PredictiveBank, candidates, choose_routes,
    compile_candidates, exposure_response, plume, reading_response)
from plume_focus.c2_r2_dynamic import ready_payload, R2Counterfactual
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.sensing_session import configuration_hash


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


class ToyWorld:
    end_time_seconds = None

    def __init__(self, cfg, seed, index, split):
        self.cfg, self.seed = cfg, seed
        rng = np.random.default_rng(seed)
        kinds = ['blank', 'old_stronger', 'old_wind'] if split == 'calibration' else ['blank', 'old_stronger', 'old_wind', 'second_north', 'second_south', 'second_weak']
        self.kind = kinds[index % len(kinds)]
        self.strength = 0. if self.kind == 'blank' else float(rng.uniform(.85, 1.15))
        self.wind = float(rng.uniform(-.025, .025))
        if self.kind == 'old_stronger':
            self.strength = float(rng.uniform(1.4, 1.8))
        if self.kind == 'old_wind':
            self.wind = float(rng.choice([-1, 1]) * rng.uniform(.08, .12))
        self.new_source = None
        self.new_strength = 0.
        if self.kind.startswith('second'):
            # Independent jitter, not transformed copies or shared noise seeds.
            y = {'second_north': .28, 'second_south': .73, 'second_weak': .36}[self.kind]
            self.new_source = (float(y + rng.uniform(-.045, .045)), float(rng.uniform(.44, .58)))
            self.new_strength = float(rng.uniform(.65, 1.2)) if self.kind != 'second_weak' else float(rng.uniform(.25, .45))
        self.event = cfg['event_seconds']

    def signal_at(self, times, positions_yx):
        old = self.strength * plume(positions_yx, self.cfg['old_source_yx'], self.wind)
        if self.new_source is not None:
            old = old + self.new_strength * plume(positions_yx, self.new_source, self.wind) * (np.asarray(times) >= self.event)
        return old

    def metadata(self):
        return dict(kind=self.kind, seed=self.seed, old_strength=self.strength, true_wind_bias=self.wind,
                    new_source=self.new_source, new_strength=self.new_strength, event_seconds=self.event)


def make_config(root, cfg):
    raw = json.loads((root / cfg['base_config']).read_text(encoding='utf-8'))
    p, m, settings = raw['session']['instrument'], raw['session']['model'], raw['session']['settings']
    p['lattice_shape'] = cfg['lattice_shape']
    p['profile_id'] = 'active_rescan_toy_8x12_v1'
    p['acquisition'].update(cadence_hz=cfg['cadence_hz'], detector_capacity_hz=cfg['cadence_hz'], integration_seconds=cfg['integration_seconds'])
    p['mechanics']['sweep_vertical_fov_per_second'] = cfg['cadence_hz'] / cfg['lattice_shape'][0]
    p['footprint'].update(height=1 / cfg['lattice_shape'][0], width=1 / cfg['lattice_shape'][1])
    m['grid_shape'] = cfg['model_shape']
    raw['session']['source_position'] = cfg['old_source_yx']
    settings['delivery_delay_seconds'] = cfg['delivery_delay_seconds']
    return EvidenceGatedRunConfig.from_dict(raw)


def natural_history(root, cfg, world):
    run = make_config(root, cfg)
    c = run.make_controller()
    plan = c.plan_next(0., run.session.home_yx, ScanRequestV1(100.), wind=CausalWind(velocity_yx=(0., .02)))
    ex = FakeScanner(world, noise_std=cfg['noise_std'], seed=world.seed + 100000).execute(plan, delivery_delay=cfg['delivery_delay_seconds'])
    trigger = None
    for r in ex.readings:
        c.ingest_readings([r], available_at_seconds=r.available_at_seconds)
        nominal = reading_response(r, cfg['old_source_yx'])
        outside = np.linalg.norm(np.asarray(r.position_yx) - cfg['old_source_yx']) >= cfg['hotspot_exclusion_radius']
        if trigger is None and outside and r.value - nominal >= cfg['trigger_residual']:
            trigger = r
    c.finish_scan(ex.receipt)
    if c.active_plan is not None or not c.startup_complete:
        raise RuntimeError('Natural FULL did not finish')
    return run, ready_payload(c, provenance={'kind': 'natural_FULL_actual_readings'}), plan, ex, trigger


def initial_bank(cfg, locations, history, trigger):
    bank = PredictiveBank(cfg, locations)
    for r in history:
        old = {w: reading_response(r, cfg['old_source_yx'], w) for w in cfg['wind_biases']}
        # Unknown onset is approximated by the first suspicious acquisition.
        # No true onset enters either planner or discriminator.
        new = {(s, w): reading_response(r, s, w) * (r.acquired_at_seconds >= trigger.acquired_at_seconds)
               for s in locations for w in cfg['wind_biases']}
        bank.update(r.reading_id, r.value, bank.means(old, new), count=False)
    return bank


def response_matrix(bank, plan, cfg, locations):
    old = {w: exposure_response(plan, cfg['old_source_yx'], w) for w in cfg['wind_biases']}
    new = {(s, w): exposure_response(plan, s, w) for s in locations for w in cfg['wind_biases']}
    return np.array([bank.means({w: v[i] for w, v in old.items()}, {k: v[i] for k, v in new.items()}) for i in range(len(plan.samples))])


def evaluate_arm(saved, record, cfg, world, initial, locations, threshold, t0, keep):
    controller = R2Counterfactual(saved)
    plan = controller.begin_record(record)
    execution = FakeScanner(world, noise_std=cfg['noise_std'], seed=world.seed + 100000).execute(plan, delivery_delay=cfg['delivery_delay_seconds'])
    bank = deepcopy(initial)
    means = response_matrix(bank, plan, cfg, locations)
    rows, cols = cfg['evaluation_shape']
    y, x = np.meshgrid((np.arange(rows) + .5) / rows, (np.arange(cols) + .5) / cols, indexing='ij')
    points = np.column_stack((y.ravel(), x.ravel()))
    hotspot = np.linalg.norm(points - cfg['old_source_yx'], axis=1) <= .18
    times = np.arange(t0, t0 + cfg['common_window_seconds'], cfg['evaluation_step_seconds']) + cfg['evaluation_step_seconds'] / 2
    loss, hotloss, trace, idx = [], [], [], 0
    confirmed = None
    max_evidence = 0.
    for at in times:
        while idx < len(execution.readings) and execution.readings[idx].available_at_seconds <= at:
            r = execution.readings[idx]
            evidence = bank.update(r.reading_id, r.value, means[idx])
            controller.ingest_readings([r], available_at_seconds=r.available_at_seconds)
            max_evidence = max(max_evidence, evidence)
            if confirmed is None and evidence >= threshold:
                confirmed = r.available_at_seconds
            trace.append(dict(reading=r.to_dict(), **bank.audit[-1]))
            idx += 1
        truth = world.signal_at(np.full(len(points), at), points)
        error = (bank.map_mean(points) - truth)**2
        loss.append(float(np.mean(error))); hotloss.append(float(np.mean(error[hotspot])))
    if idx != len(execution.readings):
        raise RuntimeError('Readings outside common evaluation window')
    controller.finish_scan(execution.receipt)
    # Explicit idle remainder, no second action even after early evidence.
    if controller.active_plan is not None:
        raise RuntimeError('Receipt did not close scan')
    return dict(confirm_at=confirmed, max_log_bf=max_evidence, final_log_bf=bank.evidence,
                mse_field=float(np.mean(loss)), mse_hotspot=float(np.mean(hotloss)),
                physical_scan_seconds=plan.duration_seconds,
                receipt_at=execution.receipt.received_at_seconds, n_readings=len(execution.readings),
                trace=trace if keep else [], receipt=execution.receipt.to_dict() if keep else None,
                evaluation_times=times.tolist() if keep else [], mse_curve=loss if keep else [])


def idle_mse(bank, cfg, world, t0):
    rows, cols = cfg['evaluation_shape']
    y, x = np.meshgrid((np.arange(rows) + .5) / rows, (np.arange(cols) + .5) / cols, indexing='ij')
    points = np.column_stack((y.ravel(), x.ravel()))
    truth = world.signal_at(np.full(len(points), t0), points)
    error = (bank.map_mean(points) - truth)**2
    hot = np.linalg.norm(points - cfg['old_source_yx'], axis=1) <= .18
    # Steady toy field throughout this window; no observations during idle.
    return float(np.mean(error)), float(np.mean(error[hot]))


def episode(root, cfg, split, index, threshold, keep=False):
    seed = cfg['splits'][split][0] + index
    world = ToyWorld(cfg, seed, index, split)
    run, saved, history_plan, history, trigger = natural_history(root, cfg, world)
    t0 = history.receipt.received_at_seconds
    state = None if trigger is None else VisibleState(t0, history.receipt.final_position_yx,
                  history.readings[-1].acquired_at_seconds, history.readings[-1].integration_seconds, trigger.position_yx)
    # Exogenous resource edge cases, fixed before outcomes, in every split.
    budget = .01 if index % 17 == 16 else cfg['action_budget_seconds']
    legal_t0 = index % 19 != 18
    planning_started = time.perf_counter()
    routes = compile_candidates(state, run.session.instrument, cfg, budget) if state and legal_t0 else {}
    compile_wall = time.perf_counter() - planning_started
    timing = dict(A=0., B=0., C=0.)
    chosen, scores = choose_routes(routes, state, cfg, timing) if state and legal_t0 else ({}, {})
    locations = candidates(state, cfg) if state else ()
    bank = initial_bank(cfg, locations, history.readings, trigger or history.readings[-1])
    idle_field, idle_hotspot = idle_mse(bank, cfg, world, t0)
    rows, details = [], {}
    for arm in cfg['arm_order']:
        route = chosen.get(arm)
        status = ('no_trigger' if trigger is None else 'no_legal_t0' if not legal_t0 else 'no_feasible_route' if route is None else 'ran')
        row = dict(episode=f'{split}_{index:02d}', split=split, base_seed=seed, arm=arm, kind=world.kind,
                   has_second=world.new_source is not None, status=status, route=route, event_at=world.event,
                   trigger_at=trigger.available_at_seconds if trigger else None, t0=t0, budget_seconds=budget,
                   confirm_at=None, confirmed=False, correct_confirmation=False, false_positive=False,
                   event_to_confirm=None, t0_to_confirm=None, restricted_delay=cfg['common_window_seconds'],
                   mse_field=idle_field, mse_hotspot=idle_hotspot, physical_scan_seconds=0., history_scan_seconds=history_plan.duration_seconds,
                   planning_compute_seconds=compile_wall + timing[arm], max_log_bf=0., final_log_bf=0., n_readings=0,
                   receipt_at=None, threshold=threshold, software_error=None,
                   initial_state_hash=configuration_hash(saved), visible_state_hash=configuration_hash(state.__dict__) if state else None)
        if status == 'ran':
            result = evaluate_arm(saved, routes[route][1], cfg, world, bank, locations, threshold, t0, keep)
            confirmed = result['confirm_at']
            row.update({k: result[k] for k in ['confirm_at', 'mse_field', 'mse_hotspot', 'physical_scan_seconds', 'max_log_bf', 'final_log_bf', 'n_readings', 'receipt_at']})
            row.update(status='confirmed' if confirmed is not None else 'unresolved', confirmed=confirmed is not None,
                       correct_confirmation=confirmed is not None and world.new_source is not None,
                       false_positive=confirmed is not None and world.new_source is None,
                       event_to_confirm=confirmed - world.event if confirmed is not None else None,
                       t0_to_confirm=confirmed - t0 if confirmed is not None else None,
                       restricted_delay=confirmed - t0 if confirmed is not None else cfg['common_window_seconds'])
            if keep:
                details[arm] = dict(route=route, plan=routes[route][0].to_dict(), result=result)
        rows.append(row)
    detail = dict(world=world.metadata(), history_plan=history_plan.to_dict(), history_readings=[r.to_dict() for r in history.readings],
                  history_receipt=history.receipt.to_dict(), trigger=trigger.to_dict() if trigger else None,
                  t0=t0, state=state.__dict__ if state else None, candidates=locations, scores=scores, arms=details) if keep else None
    return rows, detail


def summary(rows):
    result = {}
    for arm in ('A', 'B', 'C'):
        group = [r for r in rows if r['arm'] == arm]
        pos = [r for r in group if r['has_second']]
        neg = [r for r in group if not r['has_second']]
        valid = [r for r in group if r['mse_field'] is not None]
        result[arm] = dict(total=len(group), positives=len(pos), negatives=len(neg),
            correct=sum(r['correct_confirmation'] for r in pos), false_positives=sum(r['false_positive'] for r in neg),
            mean_restricted_delay_positive=float(np.mean([r['restricted_delay'] for r in pos])) if pos else None,
            mse_available_n=len(valid), mean_mse_field=float(np.mean([r['mse_field'] for r in valid])) if valid else None,
            statuses={s: sum(r['status'] == s for r in group) for s in sorted({r['status'] for r in group})})
    return result


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_example(path, detail, rows, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = dict(A='#555555', B='#D18A00', C='#087F8C')
    ax = axes[0]
    ax.scatter(.15, .5, marker='*', s=160, color='black', label='Known old source')
    if detail['world']['new_source']:
        y, x = detail['world']['new_source']; ax.scatter(x, y, marker='*', s=150, color='#9D263A', label='New source (evaluator only)')
    if detail['trigger']:
        y, x = detail['trigger']['position_yx']; ax.scatter(x, y, s=100, facecolors='none', edgecolors='red', label='Anomaly')
    for i, (arm, d) in enumerate(detail['arms'].items()):
        xy = np.array([d['plan']['start_position_yx']] + [s['end_yx'] for s in d['plan']['segments']])
        ax.plot(xy[:, 1] + i * .003, xy[:, 0], color=colors[arm], alpha=.8, label=arm + ': ' + d['route'])
    ax.set(xlim=(0, 1), ylim=(1, 0), xlabel='Normalized x', ylabel='Normalized y', title='Preselected test_03 (all unresolved)')
    ax.legend(fontsize=7, loc='lower left')
    ax = axes[1]
    for i, arm in enumerate(('A', 'B', 'C')):
        if arm in detail['arms']:
            d = detail['arms'][arm]
            times = [t['reading']['available_at_seconds'] for t in d['result']['trace']]
            ax.scatter(times, [i] * len(times), marker='|', color=colors[arm])
            ax.plot([detail['t0'], d['result']['receipt_at']], [i, i], color=colors[arm], lw=1)
            ax.scatter(d['result']['receipt_at'], i, marker='s', color=colors[arm])
        r = next(r for r in rows if r['arm'] == arm)
        ax.text(detail['t0'] + 4., i + .13, r['status'], fontsize=8)
    ax.axvline(detail['world']['event_seconds'], color='black', ls=':', label='Event (hidden)')
    if detail['trigger']:
        ax.axvline(detail['trigger']['available_at_seconds'], color='red', ls='--', label='Anomaly delivered')
    ax.axvline(detail['t0'], color='blue', ls='--', label='Legal t0 / FULL receipt')
    ax.set(yticks=[0, 1, 2], yticklabels=['A', 'B', 'C'], ylim=(-.4, 2.5), xlim=(0, detail['t0'] + 10.), xlabel='Simulation seconds', title='Fresh deliveries | and receipt squares')
    ax.legend(fontsize=7)
    ax = axes[2]
    for arm, d in detail['arms'].items():
        trace = d['result']['trace']
        ax.step([detail['t0']] + [t['reading']['available_at_seconds'] for t in trace], [0.] + [t['cumulative_log_bf'] for t in trace], where='post', color=colors[arm], label=arm)
    ax.axhline(threshold, color='black', ls='--', label='Locked threshold')
    ax.set(xlabel='Simulation seconds', ylabel='New-reading log predictive BF', title='Evidence from actual new readings')
    ax.legend(fontsize=8)
    fig.suptitle('Active discrimination rescan: paired toy physical simulation (not hardware)', fontsize=13)
    fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dev', 'study', 'demo'], default='demo')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / 'configs/active-rescan/protocol.json'
    cfg = json.loads(cfg_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = dict(command=sys.argv, python=platform.python_version(), platform=platform.platform(),
                    packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'opencv-python-headless', 'pytest', 'matplotlib']},
                    commit=git(root, 'rev-parse', 'HEAD'), dirty=git(root, 'status', '--porcelain'),
                    diff_sha256=hashlib.sha256(git(root, 'diff', 'HEAD').encode()).hexdigest(),
                    protocol_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(), protocol=cfg,
                    base_config_sha256=hashlib.sha256((root / cfg['base_config']).read_bytes()).hexdigest(),
                    source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / 'src/plume_focus').glob('*.py'))},
                    split_logic='one independent RNG seed per base; arms share that base, no mirrors/noise repeats',
                    planning_time='software wall clock separately reported; zero simulated blocking assumed',
                    output=str(args.output), status='started')
    write_json(args.output / 'RUN_MANIFEST.json', manifest)
    write_json(args.output / 'FROZEN_PROTOCOL.json', cfg)
    rows, detail = [], None
    if args.mode == 'demo':
        locked = json.loads((root / 'configs/active-rescan/locked_threshold.json').read_text())
        if locked['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('Protocol does not match calibration lock')
        threshold = locked['threshold']
        rows, detail = episode(root, cfg, 'test', 3, threshold, True)
    else:
        threshold = float(np.log(20.))
        splits = ['dev'] if args.mode == 'dev' else ['calibration', 'test']
        for split in splits:
            splitrows = []
            for i in range(cfg['splits'][split][1]):
                if time.perf_counter() - start > cfg['limits']['science_wall_seconds']:
                    raise TimeoutError('Frozen study wall limit reached; partial outputs retained')
                try:
                    rr, dd = episode(root, cfg, split, i, threshold, keep=(split == 'test' and i == 3))
                except Exception as exc:
                    write_json(args.output / f'FAILURE_{split}_{i:02d}.json', dict(error=repr(exc), split=split, index=i))
                    # Fail visibly, preserving the partial table; no replacement base.
                    raise
                splitrows.extend(rr)
                save_csv(args.output / f'{split.upper()}_PARTIAL.csv', splitrows)
                print(f'{split} {i + 1}/{cfg["splits"][split][1]}', flush=True)
                if dd:
                    detail = dd
            if split == 'calibration':
                threshold = max(float(np.log(20.)), max(r['max_log_bf'] for r in splitrows) + 1e-9)
                lock = dict(threshold=threshold, protocol_sha256=manifest['protocol_sha256'],
                            calibration_max_log_bf=max(r['max_log_bf'] for r in splitrows),
                            calibration_bases=cfg['splits']['calibration'][1], rule=cfg['threshold_rule'])
                write_json(args.output / 'LOCKED_THRESHOLD.json', lock)
                for r in splitrows:
                    r.update(threshold=threshold, confirm_at=None, confirmed=False, false_positive=False,
                             event_to_confirm=None, t0_to_confirm=None, restricted_delay=cfg['common_window_seconds'])
                    if r['status'] == 'confirmed':
                        r['status'] = 'unresolved'
            rows.extend(splitrows)
    save_csv(args.output / 'RESULTS.csv', rows)
    write_json(args.output / 'SUMMARY.json', {split: summary([r for r in rows if r['split'] == split]) for split in sorted({r['split'] for r in rows})})
    if detail is not None:
        write_json(args.output / 'EXAMPLE_TRAJECTORY.json', detail)
        plot_example(args.output / 'ACTIVE_RESCAN.png', detail, [r for r in rows if r['episode'] == 'test_03'], threshold)
    manifest.update(status='complete', elapsed_wall_seconds=time.perf_counter() - start,
                    result_sha256=hashlib.sha256((args.output / 'RESULTS.csv').read_bytes()).hexdigest())
    write_json(args.output / 'RUN_MANIFEST.json', manifest)
    print(json.dumps(dict(output=str(args.output), rows=len(rows), threshold=threshold, seconds=manifest['elapsed_wall_seconds'])))


if __name__ == '__main__':
    main()
