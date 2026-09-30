"""Declared persistent velocity uncertainty, independent of policy/session code."""

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from plume_focus.sensing_contracts import Message


@dataclass(frozen=True)
class PersistentMotionConfig(Message):
    bias_covariance_yx: tuple[tuple[float, float], tuple[float, float]]
    persistence: Literal["session_constant"]
    quadrature_order: int
    provenance: str

    def __post_init__(self):
        object.__setattr__(self, "bias_covariance_yx", tuple(tuple(float(x) for x in row) for row in self.bias_covariance_yx))
        cov = np.asarray(self.bias_covariance_yx)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=0, atol=1e-15):
            raise ValueError("Wind bias covariance must be finite, symmetric, and 2 by 2.")
        if np.linalg.eigvalsh(cov).min() < -1e-15:
            raise ValueError("Wind bias covariance must be positive semidefinite.")
        if self.persistence != "session_constant" or type(self.quadrature_order) is not int or self.quadrature_order not in (3, 5, 7):
            raise ValueError("Declare session-constant bias and quadrature order 3, 5 or 7.")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Motion uncertainty provenance is required.")

    @classmethod
    def _from_fields(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("PersistentMotionConfig requires all declared fields.")
        return super()._from_fields(value)
