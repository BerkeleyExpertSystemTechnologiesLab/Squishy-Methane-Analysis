"""Exposure, evidence and service contract checks; no performance acceptance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from plume_focus.bayes_two_stage import DesignState, compile_patch
from plume_focus.c3_active_sensing import (
    FullHistoryBank, GuardedTwoStagePlanner, ReadyContext, ServiceContract,
)
from plume_focus.c3_demo import demonstration_config
from plume_focus.active_rescan_study import make_config
from plume_focus.sensing_simulation import FakeScanner

ROOT = Path(__file__).parents[1]


class ZeroSignal:
    end_time_seconds = None

    def signal_at(self, times, positions):
        return np.zeros(len(times))


@pytest.fixture
def setup():
    cfg = demonstration_config(ROOT)
    profile = make_config(ROOT, cfg).session.instrument
    candidate = compile_patch('history', (.3, .5), DesignState(.2, (0., 0.), 0., .025),
                              profile, cfg, 100.)
    ex = FakeScanner(ZeroSignal(), noise_std=0.).execute(candidate['plan'], delivery_delay=.08)
    t0 = ex.receipt.received_at_seconds
    bank = FullHistoryBank.from_history(cfg, [(candidate['plan'], ex.readings)], t0)
    last = ex.readings[-1]
    context = ReadyContext(t0, tuple(ex.receipt.final_position_yx), last.acquired_at_seconds,
                           last.integration_seconds, gate_allows_service=True)
    contract = ServiceContract(t0 + 30., t0 + 20., 3., .25, .25)
    return cfg, profile, candidate, ex, bank, context, contract


def test_history_posterior_equals_independent_joint_gaussian(setup):
    cfg, _, candidate, ex, bank, context, _ = setup
    initial = FullHistoryBank(cfg, bank.locations, bank.onsets)
    mu = initial.means(candidate['plan'])
    y = np.array([r.value for r in ex.readings])
    loglikes = []
    for mean, variance in zip(mu, initial.var):
        cov = np.eye(len(y)) * cfg['noise_std']**2 + np.ones((len(y), len(y))) * variance
        residual = y - mean
        loglikes.append(-.5 * (len(y)*np.log(2*np.pi) + np.linalg.slogdet(cov)[1]
                              + residual @ np.linalg.solve(cov, residual)))
    logits = initial.logw + loglikes
    expected = np.exp(logits - logsumexp(logits))
    assert bank.weights() == pytest.approx(expected, abs=1e-10)
    assert bank.evidence == 0.
    assert abs(bank.history_log_odds) > 1e-6
    assert bank.label_probabilities().sum() == pytest.approx(1.)


def test_history_visibility_identity_and_exposure_guards(setup):
    cfg, _, candidate, ex, _, context, _ = setup
    with pytest.raises(ValueError, match='Future history'):
        FullHistoryBank.from_history(cfg, [(candidate['plan'], ex.readings)], .2)
    with pytest.raises(ValueError, match='Repeated historical exposure'):
        FullHistoryBank.from_history(cfg, [(candidate['plan'], ex.readings*2)], context.at)
    bad = replace(ex.readings[0], scan_id='not-this-scan')
    with pytest.raises(ValueError, match='another scan'):
        FullHistoryBank.from_history(cfg, [(candidate['plan'], [bad])], context.at)


@pytest.mark.parametrize('changes,reason', [
    ({'active_scan': True}, 'native_active_scan'),
    ({'full_obligation': True}, 'native_full_obligation'),
    ({'gate_allows_service': False}, 'native_gate_disallows'),
])
def test_native_obligations_are_not_overridden(setup, changes, reason):
    _, p, _, _, b, c, contract = setup
    service = GuardedTwoStagePlanner(b, p, contract)
    assert service.choose(replace(c, **changes)).reason == reason
    assert service.pending is None


def test_coverage_and_terminal_reservations(setup):
    _, p, _, _, b, c, contract = setup
    service = GuardedTwoStagePlanner(b, p, replace(contract, coverage_due=c.at))
    assert service.choose(c).reason == 'coverage_due'
    service = GuardedTwoStagePlanner(b, p, replace(contract, deadline=c.at+1., terminal_reserve=1.))
    assert service.choose(c).reason == 'no_paid_budget'


def test_real_two_stage_streaming_receipt_and_paid_recompile(setup):
    _, p, _, _, b, context, contract = setup
    service = GuardedTwoStagePlanner(b, p, contract)
    ids = []
    for stage in range(2):
        decision = service.choose(context)
        assert decision.reason == 'selected'
        plan = decision.candidate['plan']
        ids.append(plan.scan_id)
        assert plan.started_at_seconds == pytest.approx(context.at+.25)
        assert plan.completed_at_seconds+.08 <= contract.action_deadline+1e-9
        assert service.choose(context).reason == 'native_active_scan'
        ex = FakeScanner(ZeroSignal(), noise_std=0.).execute(plan, delivery_delay=.08)
        with pytest.raises(ValueError, match='Incomplete'):
            service.finish(ex.receipt, received_at=ex.receipt.received_at_seconds)
        with pytest.raises(ValueError, match='not yet available'):
            service.ingest(ex.readings[0], available_at=plan.started_at_seconds)
        for reading in ex.readings:
            service.ingest(reading, available_at=reading.available_at_seconds)
        with pytest.raises(ValueError, match='Repeated exposure'):
            service.ingest(ex.readings[-1], available_at=ex.readings[-1].available_at_seconds)
        with pytest.raises(ValueError, match='another scan'):
            service.finish(replace(ex.receipt, scan_id='wrong'), received_at=ex.receipt.received_at_seconds)
        context = service.finish(ex.receipt, received_at=ex.receipt.received_at_seconds)
        assert service.bank.evidence != 0.
    assert ids[0] != ids[1]
    assert service.choose(context).reason == 'two_stage_complete'


def test_second_context_cannot_reuse_old_pose_or_time(setup):
    _, p, _, _, b, context, contract = setup
    service = GuardedTwoStagePlanner(b, p, contract)
    decision = service.choose(context)
    ex = FakeScanner(ZeroSignal(), noise_std=0.).execute(decision.candidate['plan'], delivery_delay=.08)
    for r in ex.readings:
        service.ingest(r, available_at=r.available_at_seconds)
    actual = service.finish(ex.receipt, received_at=ex.receipt.received_at_seconds)
    with pytest.raises(ValueError, match='precedes'):
        service.choose(context)
    with pytest.raises(ValueError, match='pose differs'):
        service.choose(replace(actual, pose=(0.,0.)))
    assert service.choose(replace(actual, full_obligation=True)).reason == 'native_full_obligation'


def test_matched_first_is_independent_of_second_feedback_option(setup):
    _, p, _, _, b, context, contract = setup
    a = GuardedTwoStagePlanner(deepcopy(b), p, contract)
    m = GuardedTwoStagePlanner(deepcopy(b), p, contract)
    assert a.choose(context, adaptive=True).record == m.choose(context, adaptive=False).record


def test_hypothesis_budget_is_explicit(setup):
    _, p, _, _, b, _, contract = setup
    with pytest.raises(ValueError, match='Hypothesis budget'):
        GuardedTwoStagePlanner(b, p, replace(contract, max_components=1))


def test_history_is_not_changed_by_planning(setup):
    _, p, _, _, b, c, contract = setup
    before = (b.weights().copy(), b.history_log_odds, b.evidence, b.seen.copy())
    GuardedTwoStagePlanner(b, p, contract).choose(c)
    assert np.array_equal(before[0], b.weights())
    assert before[1:] == (b.history_log_odds,b.evidence,b.seen)


def test_declared_stop_does_not_override_coverage_obligation(setup):
    _, p, _, _, b, c, contract = setup
    stopped = replace(contract, posterior_entropy_stop=100.)
    service = GuardedTwoStagePlanner(b, p, stopped)
    assert service.choose(c).reason == 'declared_information_stop'
    service = GuardedTwoStagePlanner(b, p, replace(stopped, coverage_due=c.at))
    assert service.choose(c).reason == 'coverage_due'


def test_predictive_warning_waits_for_receipt_before_handoff(setup):
    _, p, _, _, b, c, contract = setup
    # A deliberately strict test limit exercises handoff, not calibration.
    service = GuardedTwoStagePlanner(b, p, replace(contract, max_predictive_nll=-1e100))
    decision = service.choose(c)
    ex = FakeScanner(ZeroSignal(), noise_std=0.).execute(decision.candidate['plan'], delivery_delay=.08)
    for r in ex.readings:
        service.ingest(r, available_at=r.available_at_seconds)
    assert service.choose(c).reason == 'native_active_scan'
    actual = service.finish(ex.receipt, received_at=ex.receipt.received_at_seconds)
    assert service.choose(actual).reason == 'model_review_required'
