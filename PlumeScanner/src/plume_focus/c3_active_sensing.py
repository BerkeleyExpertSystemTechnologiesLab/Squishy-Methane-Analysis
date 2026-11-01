"""Opt-in C3 interfaces, not a replacement for the team's native controller.

The frozen conditional-equal-H study remains in bayes_two_stage.py. This module
separates full-history design odds from new confirmation evidence and requires
explicit caller-supplied native gate, coverage and terminal contracts.
No world, true source, future observation or simulation seed enters selection.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product
import math
import time

import numpy as np
from scipy.special import logsumexp

from plume_focus.bayes_two_stage import (
    SourceOnsetBank, DesignState, compile_patch, lookahead, menu, select_second,
)
from plume_focus.c2_r2_mechanics import route_record, verify_route_record


def _finite_nonnegative(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def sample_index(plan, reading):
    """Bind a delivered reading to its actual exposure, not its list position."""
    if reading.scan_id != plan.scan_id:
        raise ValueError('Reading belongs to another scan')
    matches = [i for i, s in enumerate(plan.samples)
               if abs(s.acquired_at_seconds - reading.acquired_at_seconds) < 1e-9
               and np.allclose(s.position_yx, reading.position_yx, atol=1e-10, rtol=0)
               and abs(s.integration_seconds - reading.integration_seconds) < 1e-12
               and s.footprint == reading.footprint]
    if len(matches) != 1:
        raise ValueError('Reading does not identify one declared exposure')
    return matches[0]


class FullHistoryBank(SourceOnsetBank):
    """Full-history design posterior; evidence is only post-READY log BF.

    Historical and new odds add exactly once. This differs deliberately from
    the frozen study's equal H odds after conditioning within each H.
    """
    def __init__(self, cfg, locations, onsets):
        if cfg['noise_std'] <= 0 or not math.isfinite(cfg['noise_std']):
            raise ValueError('Positive finite observation noise required')
        super().__init__(deepcopy(cfg), locations, onsets)
        self.history_log_odds = 0.
        self.ready_at = None
        self.latest_available = 0.

    def weights(self):
        logits = self.logw + self.h * (self.history_log_odds + self.evidence)
        return np.exp(logits - logsumexp(logits))

    @classmethod
    def from_history(cls, cfg, packets, ready_at):
        _finite_nonnegative('READY time', ready_at)
        locations = tuple(product(cfg['source_grid_y'], cfg['source_grid_x']))
        onsets = np.linspace(0., ready_at, cfg['onset_grid_count'])
        bank = cls(cfg, locations, onsets)
        pending = []
        slots = set()
        for plan, readings in packets:
            means = bank.means(plan)
            for reading in readings:
                if reading.available_at_seconds > ready_at + 1e-10:
                    raise ValueError('Future history is not visible at READY')
                i = sample_index(plan, reading)
                slot = (plan.scan_id, i)
                if slot in slots:
                    raise ValueError('Repeated historical exposure')
                slots.add(slot)
                pending.append((reading, means[:, i]))
        for reading, means in sorted(pending, key=lambda x: (x[0].available_at_seconds, x[0].reading_id)):
            bank.update(reading.reading_id, reading.value, means)
            bank.latest_available = reading.available_at_seconds
        bank.history_log_odds = bank.evidence
        bank.evidence = 0.
        bank.ready_at = float(ready_at)
        return bank

    def label_probabilities(self):
        q = np.bincount(self.class_id, weights=self.weights(), minlength=self.nclass)
        return q / q.sum()


@dataclass(frozen=True)
class ReadyContext:
    at: float
    pose: tuple
    last_acquired: float
    last_integration: float
    # These are receiver assertions, not a new FOCUS grant from this planner.
    gate_allows_service: bool = False
    active_scan: bool = False
    full_obligation: bool = False


@dataclass(frozen=True)
class ServiceContract:
    deadline: float
    coverage_due: float
    terminal_reserve: float
    first_planning_fee: float
    second_planning_fee: float
    # A declared bounded fee in model seconds, not a measured hardware delay.
    planner_wall_limit: float = 5.
    max_candidates: int = 16
    max_components: int = 5000
    max_forecast_draws: int = 16
    posterior_entropy_stop: object = None
    max_predictive_nll: object = None

    def __post_init__(self):
        for key in ('deadline', 'coverage_due', 'terminal_reserve',
                    'first_planning_fee', 'second_planning_fee', 'planner_wall_limit'):
            _finite_nonnegative(key, getattr(self, key))
        for key in ('max_candidates', 'max_components', 'max_forecast_draws'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('Planner count limits must be positive integers')
        if self.posterior_entropy_stop is not None:
            _finite_nonnegative('posterior entropy stop', self.posterior_entropy_stop)
        if self.max_predictive_nll is not None and not math.isfinite(self.max_predictive_nll):
            raise ValueError('Predictive NLL limit must be finite when declared')

    @property
    def action_deadline(self):
        return min(self.deadline - self.terminal_reserve, self.coverage_due)


@dataclass(frozen=True)
class Selection:
    reason: str
    candidate: object = None
    record: object = None
    charged_fee: float = 0.
    planner_wall_seconds: float = 0.


class GuardedTwoStagePlanner:
    """A bounded research service with streaming ingest and receipt-gated replan.

    Native controller integration is pending. A refusal returns a reason for the
    caller to hand control back; it does not itself execute FULL or return HOME.
    The terminal reserve is caller supplied and not computed/guaranteed here.
    """
    def __init__(self, bank, profile, contract):
        if bank.ready_at is None:
            raise ValueError('A conditioned READY bank is required')
        self.bank = bank
        self.profile = profile
        self.contract = contract
        self.cfg = deepcopy(bank.cfg)
        if len(bank.h) > contract.max_components:
            raise ValueError('Hypothesis budget exceeded')
        if not 1 <= self.cfg['forecast_draws'] <= contract.max_forecast_draws:
            raise ValueError('Forecast budget exceeded')
        self.pending = None
        self.pending_slots = set()
        self.completed = 0
        self.ready_after = bank.ready_at
        self.last_context = None
        self.committed_second = None
        self.audit = []
        self.model_review_required = False

    def _guard(self, context):
        if self.pending is not None or context.active_scan:
            return 'native_active_scan'
        if context.full_obligation:
            return 'native_full_obligation'
        if not context.gate_allows_service:
            return 'native_gate_disallows'
        _finite_nonnegative('decision time', context.at)
        if context.at < max(self.ready_after, self.bank.latest_available) - 1e-9:
            raise ValueError('Decision precedes a required receipt or visible state')
        if self.last_context is not None:
            last = self.last_context
            if not np.allclose(context.pose, last.pose, atol=1e-10, rtol=0):
                raise ValueError('Actual receipt pose differs from READY context')
            if (context.last_acquired, context.last_integration) != (last.last_acquired, last.last_integration):
                raise ValueError('Actual last exposure differs from READY context')
        if context.at >= self.contract.coverage_due - 1e-9:
            return 'coverage_due'
        if self.model_review_required:
            return 'model_review_required'
        if self.completed >= 2:
            return 'two_stage_complete'
        return None

    def choose(self, context, *, adaptive=True):
        reason = self._guard(context)
        if reason:
            return Selection(reason)
        if self.contract.posterior_entropy_stop is not None:
            q = self.bank.label_probabilities()
            entropy = -float(np.sum(q * np.log(np.maximum(q, 1e-300))))
            if entropy <= self.contract.posterior_entropy_stop:
                return Selection('declared_information_stop')
        fee = self.contract.first_planning_fee if self.completed == 0 else self.contract.second_planning_fee
        state = DesignState(context.at + fee, context.pose, context.last_acquired, context.last_integration)
        if state.at >= self.contract.action_deadline:
            return Selection('no_paid_budget', charged_fee=fee)
        start = time.perf_counter()
        choices = menu(state, self.profile, self.cfg, self.contract.action_deadline)
        if len(choices) > self.contract.max_candidates:
            return Selection('candidate_budget_exceeded', charged_fee=fee)
        if not choices:
            return Selection('no_feasible_action', charged_fee=fee)
        if self.completed == 0:
            plans, tree = lookahead(self.bank, state, self.profile, self.cfg,
                                    self.contract.action_deadline, self.contract.second_planning_fee)
            if plans:
                candidate = plans['A'][0]
                self.committed_second = plans['M'][1]
            else:
                candidate, _ = select_second(self.bank, choices, self.cfg)
            self.audit.append(dict(stage=0, tree=tree))
        elif adaptive:
            candidate, scores = select_second(self.bank, choices, self.cfg)
            self.audit.append(dict(stage=1, conditional_entropies=scores))
        elif self.committed_second is not None:
            old = self.committed_second
            candidate = compile_patch(old['name'], old['target'], state, self.profile,
                                      self.cfg, self.contract.action_deadline)
        else:
            return Selection('no_committed_second', charged_fee=fee)
        wall = time.perf_counter() - start
        if wall > self.contract.planner_wall_limit:
            return Selection('planner_timeout', charged_fee=fee, planner_wall_seconds=wall)
        if candidate is None:
            return Selection('no_feasible_action', charged_fee=fee, planner_wall_seconds=wall)
        record = route_record(candidate['plan'], candidate['mask'], self.profile,
                              candidate['kwargs'], candidate['order'], candidate['down'])
        verify_route_record(record)
        self.pending = candidate
        self.pending_slots = set()
        self.last_context = context
        return Selection('selected', candidate, record, fee, wall)

    def ingest(self, reading, *, available_at):
        if self.pending is None:
            raise ValueError('No pending scan')
        if available_at < max(reading.available_at_seconds, self.bank.latest_available) - 1e-10:
            raise ValueError('Reading is not yet available or ingest is noncausal')
        plan = self.pending['plan']
        i = sample_index(plan, reading)
        if i in self.pending_slots:
            raise ValueError('Repeated exposure')
        nll = self.bank.update(reading.reading_id, reading.value, self.bank.means(plan)[:, i])
        if self.contract.max_predictive_nll is not None and nll > self.contract.max_predictive_nll:
            self.model_review_required = True
        # A model warning never cancels the already committed physical scan.
        self.bank.latest_available = float(available_at)
        self.pending_slots.add(i)

    def finish(self, receipt, *, received_at):
        if self.pending is None:
            raise ValueError('No pending scan')
        plan = self.pending['plan']
        if receipt.scan_id != plan.scan_id:
            raise ValueError('Receipt belongs to another scan')
        if received_at < max(receipt.received_at_seconds, self.bank.latest_available) - 1e-10:
            raise ValueError('Receipt is not yet available')
        if (receipt.completed_at_seconds < plan.completed_at_seconds - 1e-9
                or len(self.pending_slots) != len(plan.samples)):
            raise ValueError('Incomplete receipt or missing readings')
        if not np.allclose(receipt.final_position_yx, plan.final_position_yx, atol=1e-10, rtol=0):
            raise ValueError('Receipt final pose mismatch')
        sample = max(plan.samples, key=lambda s: s.acquired_at_seconds)
        context = replace(self.last_context, at=float(received_at), pose=tuple(receipt.final_position_yx),
                          last_acquired=sample.acquired_at_seconds, last_integration=sample.integration_seconds)
        self.ready_after = float(received_at)
        self.last_context = context
        self.pending = None
        self.pending_slots = set()
        self.completed += 1
        return context
