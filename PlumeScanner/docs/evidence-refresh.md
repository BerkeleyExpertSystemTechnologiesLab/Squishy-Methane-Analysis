# Requesting fresh measured evidence

`EvidenceRefreshScanner` is an opt-in extension of `EvidenceGatedScanner`.
A software or simulation consumer can ask for newly acquired measurements in
specified physical cells by a deadline. It receives the readings accepted by
the controller, their acceptance times and any cells that still lack evidence.

An explicit request can advance a native return-home and FULL scan at a READY
boundary. It can therefore obtain observations of a cell that the next normal
FOCUS would leave unvisited. The request is useful when the consumer has a
specific decision and needs an observation before it can make that decision.
The [B1 adapter](source-activity-refresh.md) uses this interface to request a
reading in the physical cell containing a declared source.

## Call from the execution adapter

Construct `EvidenceRefreshScanner(session, delivery="stream")` with an existing
`EvidenceGatedSession` and the adapter's declared delivery behavior. Use
`delivery="batch"` if readings are submitted at the receipt. The adapter owns the
coordinate mapping, session clock and measured timing inputs.

```python
from plume_focus.evidence_refresh import EvidenceRefreshRequest, EvidenceRefreshScanner
from plume_focus.sensing_messages_v2 import ScanRequestV1

scanner = EvidenceRefreshScanner(session, delivery=adapter_delivery_mode)

request = EvidenceRefreshRequest(
    request_id=caller_request_id,
    instrument_hash=session.instrument.digest,
    cells=caller_physical_cell_ids,
    fresh_since_seconds=caller_freshness_cutoff,
    deadline_seconds=caller_evidence_deadline,
)
decision = scanner.plan_evidence_refresh(
    request, now_seconds, current_position,
    ScanRequestV1(remaining_session_seconds),
    preparation_seconds=declared_preparation_allowance,
)
```

If the outcome is `scheduled`, submit `decision.plan` through the same adapter
that executes a plan returned by `plan_next`. Use the plan's start time. Continue
to call `scanner.ingest_readings` with the actual acceptance clock and
`scanner.finish_scan` with the actual execution receipt. An adapter that cannot
follow the declared start and delivery assumptions must handle the failure
using its existing execution and receipt contract. Poll the request with
`scanner.evidence_status(request, now_seconds=observation_clock)` using a time at
or after the controller's latest accepted event or opened plan start.

An outcome without a plan leaves the normal scan loop responsible for the next
action. Let the active scan finish if one is open. At READY the caller can ask
the ordinary `plan_next` for its next path using the remaining session time.

`cells` is a tuple of zero-based row-major indices on the physical scanner
lattice. Match the observed exposure center to that lattice. This is a request
for measured scalars per cell. The full extent of a plume and pointwise values
throughout an exposure footprint are different quantities. The whole exposure
must start at or after the freshness cutoff. Arrival means accepted at
`ingest_readings`, so a reading buffered until after the deadline cannot meet it.

## How the request resolves

| Outcome | Meaning for the caller |
| --- | --- |
| `satisfied` | Every cell has an accepted reading within the requested window. Reuse that evidence. |
| `scan_in_progress` | Wait for the current action's readings and receipt. Poll the evidence status. |
| `scheduled` | Execute the returned FULL plan. The estimate comes from the compiled path and declared delivery delay. |
| `full_scan_exceeds_budget` | The entire return, exposure, delivery and receipt cannot fit after the preparation allowance. |
| `current_full_lacks_cells` | The remaining part of an outstanding FULL would leave at least one requested cell unobserved in the window. |
| `window_unreachable` | The candidate FULL lacks timely observations for at least one cell. |
| `deadline_missed` | At least one requested cell lacks accepted evidence within the window. |

Scope completion and scan completion are distinct events. The caller can use a
streamed measurement once accepted. The controller continues the committed path
until the execution receipt and retains any incomplete FULL obligation. Focus
entry and renewal still come from the native gate's measurement evidence.

## Cost and integration limits

The candidate uses the native geometry and charges return home, movement,
settling, detector recovery and acquisition through the physical compiler. It
also charges the declared preparation allowance, delivery delay and completion
overrun. The caller supplies the remaining session duration and preparation
allowance. Compare the request's observation deadline with the planned arrival
time and the session budget with the complete FULL. This v1 uses a whole-field
path for a new request, which can be too costly for tight windows. Use the
outcome and duration to decide whether the request is worth scheduling.

Arrival estimates depend on the adapter's declared timing. Hardware command,
settle, exposure and delivery timestamps are needed to establish real timing.
The implementation is in-memory for a single live session with serialized API
calls. The built-in snapshot/replay adapter currently handles its original
controller types. A receiver should record the request call, returned plan,
actual accepted readings and receipt for a future request-aware replay.
`SensingController.from_session` continues to construct the base controller.
Use the explicit constructor above to opt in.

## Evidence to hand over

Use one actual receiver request with the region, reason for freshness, deadline,
the adapter's timing record and the action taken on the returned evidence. This
gives the project a concrete basis for deciding whether a FULL request is useful
and when it needs a cheaper confirmation path. The implementation here has
engineering checks. Receiver use, hardware timing and performance remain open.
