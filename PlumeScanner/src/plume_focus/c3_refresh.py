"""Opt-in C3 midterm v1. No world, event, truth or filesystem dependencies.

The old block selector is ported, not claimed as a new algorithm. A REFRESH
breaks the preceding-action FOCUS evidence chain and requires native FULL entry.
"""
from dataclasses import fields
from collections import deque
import numpy as np
from plume_focus.evidence_gated_loop import EvidenceGatedScanner
from plume_focus.full_fov_loop import FullFovPlan
from plume_focus.physical_scan import PhysicalPlan, compile_physical_scan
from plume_focus.sensing_policy_v2 import survey_blocks

POLICY_ID = 'c3_periodic_refresh_midterm_v1'

def clear_focus_after_refresh(scanner: EvidenceGatedScanner) -> None:
    """C3 local measurements leave entry to the next completed native FULL."""
    scanner._gate_focus_allowed = False
    scanner._gate_focus_evidence = None

class ShadowAlarm:
    """Two distinct delivered exposure slots in one fixed block within 1 s.

    Raw identity-calibrated entry level .15, refractory 2 s per block. This is
    a signal demonstration, not a confidence bound or source-identification test.
    """
    def __init__(self, shape, threshold=.15):
        self.shape=shape; self.threshold=threshold
        self.blocks=survey_blocks(shape,.2); self.seen=set(); self.hits={}; self.last={}; self.alerts=[]
    def block_of(self, position):
        r,c=np.minimum((np.asarray(position)*self.shape).astype(int),np.array(self.shape)-1)
        return next(i for i,(lo,hi,col) in enumerate(self.blocks) if lo<=r<hi and col==c)
    def accept(self, readings, at, mode):
        new=[]
        for r in readings:
            if r.reading_id in self.seen: continue
            self.seen.add(r.reading_id)
            if r.available_at_seconds>at+1e-10: raise ValueError('Unavailable alarm evidence')
            b=self.block_of(r.position_yx)
            q=self.hits.setdefault(b,deque())
            while q and q[0][1]<at-1.-1e-10: q.popleft()
            if r.value>=self.threshold:
                q.append((r.reading_id,float(at),r.acquired_at_seconds))
                if len(q)>=2 and at-self.last.get(b,-float('inf'))>=2.-1e-10:
                    a=dict(block=b,at=float(at),reading_ids=[q[-2][0],q[-1][0]],mode=mode)
                    self.alerts.append(a);new.append(a);self.last[b]=at
        return new

class RefreshScanner(EvidenceGatedScanner):
    def __init__(self, session, *, kind='sector', wait_seconds=1., h0_cells=()):
        super().__init__(session)
        if kind not in ('full','sector','disabled'): raise ValueError(kind)
        if wait_seconds<=0: raise ValueError('Positive frozen wait required')
        self.kind=kind;self.wait_seconds=float(wait_seconds);self.h0_cells=set(h0_cells)
        self.alarm=ShadowAlarm(self.capabilities.lattice_shape,session.settings.entry_level)
        self.delivered_age=np.full(self.capabilities.lattice_shape,-np.inf)
        self.due=None;self.confirm_pending=False;self.service_log=[];self.refresh_receipts=[]
    def ingest_readings(self, readings, *, available_at_seconds):
        fresh=[r for r in readings if r.reading_id not in self._seen and r.reading_id not in self._pending]
        out=super().ingest_readings(readings,available_at_seconds=available_at_seconds)
        for r in fresh:
            ij=np.minimum((np.asarray(r.position_yx)*self.capabilities.lattice_shape).astype(int),np.array(self.capabilities.lattice_shape)-1)
            self.delivered_age[tuple(ij)]=max(self.delivered_age[tuple(ij)],r.acquired_at_seconds)
        alerts=self.alarm.accept(fresh,available_at_seconds,self._active.mode if self._active else 'NONE')
        if self.kind!='disabled':
            for a in alerts:
                lo,hi,col=self.alarm.blocks[a['block']]
                if any(r*self.capabilities.lattice_shape[1]+col not in self.h0_cells for r in range(lo,hi)):
                    self.confirm_pending=True
        return out
    def _on_plan_opened(self,plan):
        if plan.mode=='REFRESH':
            self._gate_open_action=None
        else: super()._on_plan_opened(plan)
    def _on_scan_finished(self,receipt):
        if self._active.mode=='REFRESH':
            # Last-action renewal is unavailable. No FULL cells or gate evidence
            # are cleared/created, even on a partial receipt.
            clear_focus_after_refresh(self)
            self.refresh_receipts.append(dict(at=receipt.received_at_seconds,delivered=len(self._pending),scheduled=len(self._active.samples)))
            complete=(len(self._pending)==len(self._active.samples) and receipt.completed_at_seconds>=self._active.completed_at_seconds-1e-9)
            if complete:self.due=receipt.received_at_seconds+self.wait_seconds
        else:
            super()._on_scan_finished(receipt)
            if self._active.mode=='FULL' and self._full_missing is None:
                self.due=receipt.received_at_seconds+self.wait_seconds
                self.confirm_pending=False
    def plan_next(self, now_seconds,current_position,constraints,*,wind=None):
        # Call native validation/transition exactly when no service is requested.
        if self.kind=='disabled' or self._active is not None or self._full_missing is not None or not self.gate_focus_allowed:
            return super().plan_next(now_seconds,current_position,constraints,wind=wind)
        due=self.due is not None and now_seconds>=self.due-1e-10
        if not due and not self.confirm_pending:
            return super().plan_next(now_seconds,current_position,constraints,wind=wind)
        if self.confirm_pending or self.kind=='full':
            self._gate_focus_allowed=False
            p=super().plan_next(now_seconds,current_position,constraints,wind=wind)
            self.service_log.append(dict(at=now_seconds,reason='confirm' if self.confirm_pending else 'periodic_full',mode=p.mode))
            return p
        # Native zero-budget call is not used: it would create a FULL obligation.
        if now_seconds<max(self._last_available,self._belief.time_seconds)-1e-10:raise ValueError('Noncausal decision')
        if not np.allclose(current_position,self._last_pose,atol=1e-10,rtol=0):raise ValueError('Pose mismatch')
        if wind is not None and not self._wind.available_at_seconds<=wind.available_at_seconds<=now_seconds:raise ValueError('Wind must be causal')
        self._belief.advance(now_seconds,self._wind,self.source_position)
        if wind is not None:self._wind=wind
        self._last_available=now_seconds
        shape=self.capabilities.lattice_shape
        blocks=survey_blocks(shape,.2)
        idx=min(range(len(blocks)),key=lambda i:(float(np.min(self.delivered_age[blocks[i][0]:blocks[i][1],blocks[i][2]])),(i-self._rotation)%len(blocks)))
        lo,hi,col=blocks[idx];mask=np.zeros(shape,bool);mask[lo:hi,col]=True
        last=self._last_detector_reading
        def compile(m):
            return compile_physical_scan(m,self.session.instrument,scan_id=f'scan-{self._scan_count:06d}',started_at_seconds=now_seconds,current_position=tuple(current_position),last_reading_at_seconds=last.acquired_at_seconds if last else None,last_integration_seconds=last.integration_seconds if last else None)
        path=compile(mask);needed=path.duration_seconds+self.session.settings.delivery_delay_seconds+self.session.settings.completion_overrun_seconds
        fits=needed<=constraints.max_duration_seconds+1e-9
        if not fits:path=compile(np.zeros(shape,bool))
        base=FullFovPlan(**{f.name:getattr(path,f.name) for f in fields(PhysicalPlan)},mode='REFRESH' if fits else 'NONE',reason='oldest_block' if fits else 'refresh_budget_pending',gate_max_mean=float(np.max(self.predictive_moments(now_seconds,shape)[0])),threshold=self.session.settings.threshold,eligible_cells=(),omitted_cells=(),full_epoch=self._epoch,return_home_until_seconds=now_seconds,required_duration_seconds=needed)
        plan=self._decorate_plan(base,())
        if fits:self._open_scan(plan);self._on_plan_opened(plan)
        self.service_log.append(dict(at=now_seconds,reason=plan.reason,mode=plan.mode,block=idx,due=self.due))
        return plan
