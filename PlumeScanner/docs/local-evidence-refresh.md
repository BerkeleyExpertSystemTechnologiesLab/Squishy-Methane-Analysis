# A single-cell C3 refresh for B1

Use the explicit `LocalSourceActivityRefreshScanner` candidate with an
`EvidenceGatedSession` and a declared source. It can serve the existing
fresh-evidence request by compiling one physical cell as a C3 `REFRESH`.
The FULL-only entry point is also available on the candidate.

```python
from plume_focus.local_evidence_refresh import LocalSourceActivityRefreshScanner
from plume_focus.sensing_messages_v2 import ScanRequestV1

scanner = LocalSourceActivityRefreshScanner(session, delivery=adapter_delivery)
# The adapter executes and delivers the normal startup scan first.
request = scanner.source_activity_request(
    "source-check", fresh_since_seconds=event_time, deadline_seconds=decision_deadline,
)
decision = scanner.plan_local_evidence_refresh(
    request, now, current_pose, ScanRequestV1(remaining_seconds),
    terminal="home_full", preparation_seconds=planning_allowance,
)
```

The caller chooses `terminal="receipt"` when a completed measurement action
is sufficient. Use `terminal="home_full"` when the session also owes a
complete native return to HOME and FULL. The latter reserves a subsequent
action from the local plan's estimated final pose and exposure. The allowance
is charged for each opened action. At the actual local receipt, use its
time and position to plan the required FULL under the remaining budget.
A satisfied evidence request may simply reuse a reading, so the caller
continues to track its mission terminal independently of the evidence status.

The selector checks a measurement deadline and total action budget
separately. A local action is eligible only after a completed startup FULL,
at a READY boundary, with no outstanding FULL coverage and with native
FOCUS permission. If both options deliver the requested evidence on time,
choose the one with lower estimated cost to the declared terminal. When
the original FULL is feasible and the local action is unavailable,
`local_reason` describes the fallback. The controller accounts for stream or
batch delivery and checks fulfillment against actual acceptance times.

Dispatch the chosen plan through the same execution adapter. Feed actual
`SensorReading` messages to `ingest_readings` and the actual
`ExecutionReceipt` to `finish_scan`. The same ingress updates B1 and records
evidence for the request. A local receipt clears FOCUS permission under the
existing C3 rule. It does not credit any native FULL cells. The next FULL
evaluates entry with its own delivered measurements.

This candidate uses the C3 prototype's typed REFRESH and physical compiler.
The base factory continues to serve the registered controller types. The
prototype is single-session, with hardware acceptance and candidate-specific
snapshot restore open. Timing quotes depend on the instrument profile and
adapter delays. The current source-activity belief is B1 V1.
