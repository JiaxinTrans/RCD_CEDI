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
    """SI vehicle input: scalar speed and velocity heading in radians.

    ``vx`` and ``vy`` are read-only components derived from speed/heading.
    The rectangle faces the same heading unless keyword-only ``body_heading``
    is supplied, e.g. for a measured body/motion misalignment. Body orientation
    is fixed during prediction. IDs must be exactly representable integers.
    """

    vid: int
    x: float
    y: float
    speed: float
    heading: float
    length: float
    width: float
    body_heading: float | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if isinstance(self.vid, bool) or not isinstance(self.vid, Integral) or abs(self.vid) > 2**53:
            raise ValueError('vid must be an integer with abs(vid) <= 2**53')
        object.__setattr__(self, 'vid', int(self.vid))
        for name in ('x', 'y', 'speed', 'heading', 'length', 'width'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if self.speed < 0:
            raise ValueError('speed must be nonnegative; use heading for direction')
        if self.body_heading is not None:
            angle = float(self.body_heading)
            if not math.isfinite(angle):
                raise ValueError('body_heading must be finite when supplied')
            object.__setattr__(self, 'body_heading', angle)
        if self.length <= 0 or self.width <= 0:
            raise ValueError('vehicle dimensions must be positive')

    @cached_property
    def _velocity(self):
        return (self.speed * math.cos(self.heading), self.speed * math.sin(self.heading))

    @property
    def vx(self):
        """Derived x velocity, m/s; not a constructor input."""
        return self._velocity[0]

    @property
    def vy(self):
        """Derived y velocity, m/s; not a constructor input."""
        return self._velocity[1]

    @property
    def body_angle(self):
        """Effective rectangle angle; defaults to the velocity heading."""
        return self.heading if self.body_heading is None else self.body_heading

    @cached_property
    def basis(self):
        c, s = math.cos(self.body_angle), math.sin(self.body_angle)
        return ((c, s), (-s, c))

    @cached_property
    def direction(self):
        return (self.vx / self.speed, self.vy / self.speed) if self.speed > EPS else self.basis[0]

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
        return self.body_angle


def _at_motion(state, motion):
    """Internal audit adapter: retain body orientation after a control action.

    Motion kernels use Cartesian components. Reconstruct the public polar
    representation without rotating the predicted fixed-heading rectangle.
    This helper is not an additional public input mode.
    """
    speed = math.hypot(motion[2], motion[3])
    heading = math.atan2(motion[3], motion[2]) if speed > 0 else state.heading
    return VehicleState(state.vid, motion[0], motion[1], speed, heading,
                        state.length, state.width, body_heading=state.body_angle)


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
