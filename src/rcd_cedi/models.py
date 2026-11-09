"""Public SI state and result types shared by all three entry points."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from functools import cached_property
from numbers import Integral
from typing import Any

from .constants import EPS


@dataclass(frozen=True)
class VehicleState:
    """Fixed-heading rectangle: metres, seconds and heading in radians.

    Heading describes the body; velocity need not be aligned with it.
    IDs are unique integers representable exactly in the numerical kernels.
    """

    vid: int
    x: float
    y: float
    vx: float
    vy: float
    heading: float
    length: float
    width: float

    def __post_init__(self):
        if isinstance(self.vid, bool) or not isinstance(self.vid, Integral) or abs(self.vid) > 2**53:
            raise ValueError('vid must be an integer with abs(vid) <= 2**53')
        object.__setattr__(self, 'vid', int(self.vid))
        for name in ('x', 'y', 'vx', 'vy', 'heading', 'length', 'width'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if self.length <= 0 or self.width <= 0:
            raise ValueError('vehicle dimensions must be positive')

    @cached_property
    def basis(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return ((c, s), (-s, c))

    @cached_property
    def direction(self):
        speed = math.hypot(self.vx, self.vy)
        return (self.vx / speed, self.vy / speed) if speed > EPS else self.basis[0]

    def radius(self, n):
        return (self.length / 2 * abs(n[0] * self.basis[0][0] + n[1] * self.basis[0][1])
                + self.width / 2 * abs(n[0] * self.basis[1][0] + n[1] * self.basis[1][1]))

    # Read-only Frenet aliases retain the supplied MLC equations unchanged.
    @property
    def s(self):
        return self.x

    @property
    def d(self):
        return self.y

    @property
    def vs(self):
        return self.vx

    @property
    def vd(self):
        return self.vy

    @property
    def theta(self):
        return self.heading


@dataclass(frozen=True)
class RCDResult:
    status: str
    subject_vehicle: int
    opponent_vehicle: int
    predicted_contact: bool
    reachable: bool | None
    initial_et: float | None
    blockers: tuple[int, ...] = ()
    unresolved_vehicles: tuple[int, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CEDIResult:
    """CEDI plus native solver diagnostics; None denotes an unscored state."""

    status: str
    cedi: float | None
    subject_vehicle: int
    opponent_vehicle: int | None
    initial_et: float | None
    ax: float | None = None
    ay: float | None = None
    front_demand: float | None = None
    rear_demand: float | None = None
    responses: tuple[dict[str, Any], ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
