from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Metrics:
    detection_time_s: Optional[float]
    full_scan_time_s: Optional[float]
    total_time_s: float
    total_motion_s: float
    total_dwell_s: float
    steps: int
