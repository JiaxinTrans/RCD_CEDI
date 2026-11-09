"""User-configurable CEDI settings; distances in m, times in s,
Road/work-zone defaults describe a generic example, not a study site.
Pass keyword arguments to MLCConfig or VehicleConfig to use another scenario.
The normalization scales for ax, ay, afs, ars (surround_brake_g), and other demand terms are user-configurable positive values.
They can be selected according to the specific scenario and assessment objectives to balance CEDI's sensitivity to different demand components.
For a given demand, a smaller scale increases its contribution to CEDI, whereas a larger  scale reduces its contribution.
The default values are used in our paper and can be adjusted by the user.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .constants import G


def _positive(name, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _steps(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError(f'{name} must be an integer >= 2')



@dataclass(frozen=True)
class VehicleConfig:
    """Vehicle CEDI settings with generic road and 30-step search defaults.

    horizon is a packed-kernel placeholder, overwritten by initial ET. It
    does not truncate the vehicle action or provide an ET acceptance filter.
    ax/ay and surrounding g scales normalize costs at every expansion level.
    """

    horizon: float = 10.0
    road_min: float = 0.0
    road_max: float = 10.5  # Example: three 3.5-m lanes; independent of lane_width.
    lane_width: float = 3.5
    gap: float = 0.01
    ax_accel_g: float = 0.20
    ax_brake_g: float = 0.75
    ay_g: float = 0.30
    surround_brake_g: float = 0.75
    ax_steps: int = 30
    ay_steps: int = 30
    brake_step: float = 0.00625
    sector_steps: int = 30
    sample_s: float = 0.1
    expansion_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)
    include_tail: bool = True
    expand_response_limits: bool = True
    audit: bool = True

    def __post_init__(self):
        for name in ('horizon', 'lane_width', 'ax_accel_g', 'ax_brake_g', 'ay_g',
                     'surround_brake_g', 'brake_step', 'sample_s'):
            _positive(name, getattr(self, name))
        for name in ('ax_steps', 'ay_steps', 'sector_steps'):
            _steps(name, getattr(self, name))
        if not math.isfinite(self.gap) or self.gap < 0:
            raise ValueError('gap must be finite and nonnegative')
        if not (math.isfinite(self.road_min) and math.isfinite(self.road_max)
                and self.road_min < self.road_max):
            raise ValueError('road bounds must be finite and increasing')
        factors = tuple(float(f) for f in self.expansion_factors)
        if (not factors or factors[0] != 1.0
                or any(not math.isfinite(f) or f <= 0 for f in factors)
                or any(a >= b for a, b in zip(factors, factors[1:]))):
            raise ValueError('expansion_factors must start at 1 and strictly increase')
        object.__setattr__(self, 'expansion_factors', factors)

    def numeric(self):
        return np.asarray([self.horizon, self.road_min, self.road_max, self.gap,
                           self.lane_width, self.ax_accel_g * G, self.ax_brake_g * G,
                           self.ay_g * G, self.surround_brake_g * G], dtype=float)


@dataclass(frozen=True)
class MLCConfig:
    """Generic example boundary and configurable first-feasible-stage search.

    The supplied boundary equations assume zone_start[1] == 0. This is
    explicitly enforced rather than silently accepting a shifted boundary.
    ay limits remain reference scales, as in the supplied MLC calculation;
    they do not impose a new rejection threshold on InDepth/ET demand.
    """

    # Example 100-m taper closing the first of three 3.5-m lanes.
    # Boundary points, lane centres and road bounds are independent inputs:
    # when changing the road layout, configure all related values together.
    zone_start: tuple[float, float] = (0.0, 0.0)
    zone_end: tuple[float, float] = (100.0, 3.5)
    lane_width: float = 3.5
    lane_origin: float = 1.75  # Centre of lane 0, not a road-edge coordinate.
    target_lane_offset: int = 1  # Target lane ID minus original lane ID.
    road_min_d: float = 0.0
    road_max_d: float = 10.5
    gap: float = 0.01
    horizon: float = 6.0
    tau_steps: int = 30
    ax_steps: int = 30
    # Keep the source acceleration scales; callers may override them.
    ax_brake_limits_g: tuple[float, float] = (0.75, 1.50)
    ax_accel_limits_g: tuple[float, float] = (0.20, 0.40)
    ay_limits_g: tuple[float, float] = (0.30, 0.60)
    surrounding_brake_limits_g: tuple[float, float] = (0.30, 0.60)

    def __post_init__(self):
        for name in ('lane_width', 'horizon'):
            _positive(name, getattr(self, name))
        for name in ('tau_steps', 'ax_steps'):
            _steps(name, getattr(self, name))
        for name in ('ax_brake_limits_g', 'ax_accel_limits_g', 'ay_limits_g',
                     'surrounding_brake_limits_g'):
            values = tuple(float(x) for x in getattr(self, name))
            if len(values) != 2 or any(not math.isfinite(x) or x <= 0 for x in values) or values[1] < values[0]:
                raise ValueError(f'{name} must contain two positive nondecreasing scales')
            object.__setattr__(self, name, values)
        for name in ('zone_start', 'zone_end'):
            point = tuple(float(v) for v in getattr(self, name))
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError(f'{name} must be a finite 2D point')
            object.__setattr__(self, name, point)
        if self.zone_start[1] != 0:
            raise ValueError('the supplied MLC boundary model requires zone_start[1] == 0')
        if self.zone_end[0] <= self.zone_start[0]:
            raise ValueError('zone_end x must exceed zone_start x')
        if not math.isfinite(self.lane_origin):
            raise ValueError('lane_origin must be finite')
        if (isinstance(self.target_lane_offset, bool)
                or not isinstance(self.target_lane_offset, Integral)
                or self.target_lane_offset == 0):
            raise ValueError('target_lane_offset must be a nonzero integer')
        if not math.isfinite(self.gap) or self.gap < 0:
            raise ValueError('gap must be finite and nonnegative')
        if not (math.isfinite(self.road_min_d) and math.isfinite(self.road_max_d)
                and self.road_min_d < self.road_max_d):
            raise ValueError('road bounds must be finite and increasing')
