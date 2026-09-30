"""Historical allocation adapters; no changes to saved session/message semantics."""
import numpy as np
from plume_focus.sensing_session_v2 import SensingSessionV2
from plume_focus.sensing_session_v4 import SensingSessionV4
from plume_focus.sensing_session_v5 import SensingSessionV5
from plume_focus.sensing_policy_v1 import plan_scan_v1
from plume_focus.sensing_policy_v2 import plan_scan_v2, survey_blocks
from plume_focus.sensing_policy_v4 import plan_scan_v4
from plume_focus.sensing_policy_v5 import plan_scan_v5


class HistoricalPolicy:
    def __init__(self, session):
        self.session = session
        self.survey = isinstance(session, SensingSessionV2)
        self.small_service = isinstance(session, SensingSessionV4)
        self.threshold = isinstance(session, SensingSessionV5)
        self.planner = (plan_scan_v5 if self.threshold else plan_scan_v4 if self.small_service
                        else plan_scan_v2 if self.survey else plan_scan_v1)

    def plan(self, controller, request, now, pose):
        c = controller
        last = c._last_detector_reading
        extra = {"last_survey_completed": c._last_survey_completed} if self.survey else {}
        if self.threshold:
            extra["has_delivered_evidence"] = bool(c._seen)
        return self.planner(c._belief, self.session, request, c._wind, c._last_sampled,
            scan_id=f"scan-{c._scan_count:06d}", now=now, pose=pose, rotation=c._rotation,
            startup_complete=c.startup_complete,
            last_reading_at_seconds=last.acquired_at_seconds if last else None,
            last_integration_seconds=last.integration_seconds if last else None, **extra)

    def finished(self, controller, receipt):
        c = controller
        if self.small_service:
            visited = {tuple(np.minimum((np.asarray(r.position_yx)*c.capabilities.lattice_shape).astype(int),
                       np.asarray(c.capabilities.lattice_shape)-1)) for r in c._pending.values()}
            required = c._active.service_required_cells
            if required and all(divmod(cell,c.capabilities.lattice_shape[1]) in visited for cell in required):
                c._last_survey_completed = receipt.received_at_seconds
        elif self.survey and c._active.status in ("reacquisition", "startup_full_scene"):
            c._last_survey_completed = receipt.received_at_seconds
        count = (len(survey_blocks(c.capabilities.lattice_shape,self.session.policy.survey_height_fov))
                 if self.survey else c.capabilities.lattice_shape[1])
        c._rotation = (c._rotation + 1) % count
