# Request a fresh reading for a known source

`SourceActivityRefreshScanner` combines the two-model B1 belief with the
fresh-measurement request API. The caller supplies the observation window for a
decision involving a declared source. The scanner maps the source coordinate
to a physical cell and checks whether it already has a usable reading. A new
request can open a native FULL at a legal planning boundary when both the
observation deadline and the complete scan budget fit.

```python
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.source_activity_refresh import SourceActivityRefreshScanner

scanner = SourceActivityRefreshScanner(session, delivery=adapter_delivery_mode)
request = scanner.source_activity_request(
    "source-check-01",
    fresh_since_seconds=event_time,
    deadline_seconds=decision_deadline,
)
decision = scanner.plan_evidence_refresh(
    request, now, current_position, ScanRequestV1(remaining_session_seconds),
    preparation_seconds=preparation_allowance,
)
```

Use an `EvidenceGatedSession` with a declared source. Record the candidate ID
`source_activity_evidence_refresh_v1_candidate`, delivery mode, activity prior
and session in the adapter's run record. The built-in factory for the native
session still constructs its registered scanner. This candidate is selected
with the constructor above.

On `scheduled`, send the returned plan through the existing execution adapter.
Call `ingest_readings(readings, available_at_seconds=accepted_at)` on each
actual ingress and `finish_scan(receipt)` on actual completion. The accepted
reading enters the B1 comparison through that one ingress. A reused reading
is already in the belief. A valid reading received after the request deadline
can still update the current belief, while the request reports a missed
deadline. An early reading leaves the current scan open until its receipt.

```python
measured = scanner.evidence_status(request, now_seconds=adapter_clock)
models = scanner.source_activity_state()
```

`measured` reports which physical cells have timely readings and the actual
acceptance time of each reading. `models` reports the cumulative weight of the
continuing-injection model from all accepted readings, the log-odds, the belief
time and the controller time. Its weight depends on the initial prior and the
two declared models. A single reading can have little power to separate them.
The field estimate uses the mixture moments. A validated probability of source
cessation, intermittent-state tracking and unknown-source localization remain
separate research goals.

The native measurement-error gate still determines eligibility for FOCUS. An
eligible action can use the new estimated field when it selects a region. The
request may use a whole-field return and scan to obtain one reading. A shorter
confirmation path and calibrated timing would require their own contracts.
The candidate runs in one live session with serialized calls. A checkpoint
adapter for the two component beliefs and accepted-time ledger remains open.

For a receiver handoff, record a real decision, its source cell and observation
deadline, the returned plan or refusal, the actual reading acceptance, the
model weight before and after ingress, the completion receipt and what the
consumer did with the result.
