"""Rectangle geometry mapped to the manuscript, Section 3.1 / 3.2.1.

Paper map / 论文标定:
    Eq. (1): Vehicle.from_paper_state, Vehicle.axes
    Eq. (3): Vehicle.initial_corners, ordered A1,A2,A3,A4
    Eqs. (4)-(6): vertex_edge_candidates / ray_edge_candidates
    Eq. (7): closest_collision_sets -> A_c, B_c
    Eqs. (8)-(9): spatial_interaction -> P1, V_AB_c
    Eq. (15): contact_time_from_pairs

SAT is an independent numeric check and a documented completion for parallel /
collinear edges; it is not presented as the paper's ray-intersection formula.
Public VehicleState uses radians; the internal compatibility Vehicle stores
heading_deg. Use from_paper_state to follow the paper's speed-heading Eq. (1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from math import cos, hypot, inf, radians, sin
from typing import NamedTuple

import numpy as np
from numba import njit

from .constants import EPS, Vec

# Numerical tolerances, not physical reachability / risk acceptance thresholds.
PAIR_DISTANCE_RTOL = 1e-6
CONTACT_TIME_RTOL = 1e-6
_UNSET = object()


def add(a: Vec, b: Vec) -> Vec:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Vec, b: Vec) -> Vec:
    return a[0] - b[0], a[1] - b[1]


def mul(s: float, a: Vec) -> Vec:
    return s * a[0], s * a[1]


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec, b: Vec) -> float:
    """Signed 2D cross product a_x*b_y-a_y*b_x; order matters in Eq. (14)."""
    return a[0] * b[1] - a[1] * b[0]


def _finite_pair(value, name):
    pair = tuple(float(x) for x in value)
    if len(pair) != 2 or not all(math.isfinite(x) for x in pair):
        raise ValueError(f'{name} must be a finite 2D vector')
    return pair


@dataclass(frozen=True)
class Vehicle:
    """Fixed-heading rectangular actor used by the internal RCD functions.

    Eq. (1) uses the same heading for body geometry and velocity direction.
    The general constructor retains independently recorded velocity/body
    heading for compatibility; paper_aligned makes that extension observable.
    """

    name: str
    center: Vec
    velocity: Vec
    length: float
    width: float
    heading_deg: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'name', str(self.name))
        object.__setattr__(self, 'center', _finite_pair(self.center, 'center'))
        object.__setattr__(self, 'velocity', _finite_pair(self.velocity, 'velocity'))
        for name in ('length', 'width', 'heading_deg'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if self.length <= 0 or self.width <= 0:
            raise ValueError('vehicle dimensions must be positive')

    @classmethod
    def from_paper_state(cls, name, x, y, speed, heading_rad, length, width):
        """Eq. (1): P=(x,y), theta=(cos psi,sin psi), v=speed*theta."""
        if not math.isfinite(speed) or speed < 0:
            raise ValueError('scalar speed must be finite and nonnegative')
        if not math.isfinite(heading_rad):
            raise ValueError('heading_rad must be finite')
        velocity = (speed * cos(heading_rad), speed * sin(heading_rad))
        return cls(name, (x, y), velocity, length, width, math.degrees(heading_rad))

    @cached_property
    def axes(self):
        """Eq. (1), with theta_perp=(-sin psi,cos psi)."""
        angle = radians(self.heading_deg)
        return (cos(angle), sin(angle)), (-sin(angle), cos(angle))

    @property
    def paper_aligned(self):
        speed = hypot(*self.velocity)
        if speed <= EPS:
            return True
        direction = mul(1. / speed, self.velocity)
        return hypot(*sub(direction, self.axes[0])) <= 1e-7

    @cached_property
    def initial_corners(self):
        """Eq. (3): 1 front-left, 2 front-right, 3 rear-right, 4 rear-left.

        Signs are (-1)**floor((i-1)/2), (-1)**floor(i/2), i=1..4.
        Corner numbering follows the equation, not inconsistent figure labels.
        """
        forward, normal = self.axes
        return tuple(
            add(self.center, add(mul(along * self.length / 2, forward),
                                 mul(across * self.width / 2, normal)))
            for along, across in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        )

    def corners(self, t=0.):
        """Translate the Eq. (3) rectangle under CV; no body rotation."""
        if t == 0.:
            return self.initial_corners
        displacement = mul(t, self.velocity)
        return tuple(add(point, displacement) for point in self.initial_corners)


class Candidate(NamedTuple):
    """Paper's (V_i,E_i): initial corner and relative-frame intersection.

    The paragraph before Eq. (9) switches E/F notation. Here V is always the
    vertex and E is always the intersection, matching Eqs. (10),(13),(15).
    E is not the future world-coordinate contact point.
    """

    vertex_vehicle: str
    vertex_index: int
    edge_vehicle: str
    edge_index: int
    V: Vec
    E: Vec
    time: float
    edge_fraction: float | None = None

    @property
    def distance(self):
        return hypot(*sub(self.E, self.V))


@dataclass(frozen=True)
class SpatialInteraction:
    """Auditable Eqs. (7)-(9) result; initial contact is a separate state."""

    p1: bool
    contact_time: float | None
    a_c: tuple[Candidate, ...] = ()
    b_c: tuple[Candidate, ...] = ()
    pairs: tuple[Candidate, ...] = ()
    method: str = 'ray_equations_4_9'
    initial_contact: bool = False


def swept_contact(a: Vehicle, b: Vehicle) -> float | None:
    """Exact CV rectangle SAT time; numeric check / degeneracy completion.

    None: no future contact. Zero: already touching/overlapping. Positive:
    first future contact. This helper itself does not evaluate paper P2.
    """
    relative_position = sub(b.center, a.center)
    relative_velocity = sub(b.velocity, a.velocity)
    entry, exit_ = -inf, inf
    a_forward, a_left = a.axes
    b_forward, b_left = b.axes
    for axis in (a_forward, a_left, b_forward, b_left):
        radius = (
            a.length / 2 * abs(dot(a_forward, axis))
            + a.width / 2 * abs(dot(a_left, axis))
            + b.length / 2 * abs(dot(b_forward, axis))
            + b.width / 2 * abs(dot(b_left, axis))
        )
        position = dot(relative_position, axis)
        speed = dot(relative_velocity, axis)
        if abs(speed) < EPS:
            if abs(position) > radius + EPS:
                return None
            continue
        t1, t2 = (-radius - position) / speed, (radius - position) / speed
        entry = max(entry, min(t1, t2))
        exit_ = min(exit_, max(t1, t2))
        if entry > exit_ + EPS:
            return None
    return None if exit_ < -EPS else max(0., entry)


def ray_edge_candidates(moving: Vehicle, fixed: Vehicle) -> list[Candidate]:
    """Eqs. (4)-(6), one direction: all valid corner-ray / edge crossings.

    Solve V+t*v_relative=edge_start+lambda*edge. Eq. (5) excludes a zero
    denominator. Eq. (4) accepts t>=0 and lambda in [0,1], up to EPS roundoff.
    Parallel/coincident contact is completed separately by SAT.
    """
    output = []
    relative_velocity = sub(moving.velocity, fixed.velocity)
    fixed_corners = fixed.corners()
    for i, vertex in enumerate(moving.corners(), start=1):
        for j, edge_start in enumerate(fixed_corners, start=1):
            edge = sub(fixed_corners[j % 4], edge_start)
            denominator = cross(relative_velocity, edge)
            if abs(denominator) < EPS:
                continue
            delta = sub(edge_start, vertex)
            t_ij = cross(delta, edge) / denominator
            lambda_ij = cross(delta, relative_velocity) / denominator
            if t_ij >= -EPS and -EPS <= lambda_ij <= 1 + EPS:
                intersection = add(edge_start, mul(lambda_ij, edge))
                output.append(Candidate(moving.name, i, fixed.name, j,
                                        vertex, intersection, max(0., t_ij), lambda_ij))
    return output


def vertex_edge_candidates(a, b):
    """Eqs. (4)-(6), both A relative to B and B relative to A."""
    return ray_edge_candidates(a, b) + ray_edge_candidates(b, a)


def _minimum_distance_pairs(candidates):
    if not candidates:
        return ()
    shortest = min(pair.distance for pair in candidates)
    tolerance = PAIR_DISTANCE_RTOL * max(1., shortest)
    return tuple(pair for pair in candidates if abs(pair.distance - shortest) <= tolerance)


def closest_collision_sets(a, b):
    """Eq. (7): directional minimum-distance sets A_c and B_c, ties retained."""
    return (_minimum_distance_pairs(ray_edge_candidates(a, b)),
            _minimum_distance_pairs(ray_edge_candidates(b, a)))


def _on_segment(point, start, end):
    edge, rel = sub(end, start), sub(point, start)
    scale = max(1., hypot(*edge))
    return (abs(cross(rel, edge)) <= 1e-7 * scale
            and -1e-7 <= dot(rel, edge) <= dot(edge, edge) + 1e-7)


def contact_pairs_at_time(a, b, t):
    """Supplement Eq. (5)'s undefined parallel / coincident-edge cases.

    At exact SAT contact recover corner-on-edge pairs, then subtract the
    fixed vehicle displacement to express E in the same relative frame as V.
    """
    output = []
    for moving, fixed in ((a, b), (b, a)):
        fixed_at_contact = fixed.corners(t)
        for i, (initial, contact) in enumerate(zip(moving.corners(), moving.corners(t)), 1):
            for j, start in enumerate(fixed_at_contact, 1):
                end = fixed_at_contact[j % 4]
                if _on_segment(contact, start, end):
                    edge = sub(end, start)
                    fraction = dot(sub(contact, start), edge) / dot(edge, edge)
                    intersection = sub(contact, mul(t, fixed.velocity))
                    output.append(Candidate(moving.name, i, fixed.name, j,
                                            initial, intersection, t, fraction))
    return output


def contact_time_from_pairs(pairs, relative_speed):
    """Eq. (15): t_AB_c=||E_i-V_i||/||v_AB|| for a selected closest pair."""
    if not pairs or relative_speed <= EPS:
        raise ValueError('Eq. (15) requires selected pairs and nonzero relative speed')
    return min(pair.distance for pair in pairs) / relative_speed


def spatial_interaction(a, b, contact_time=_UNSET):
    """Eqs. (7)-(9), cross-checked by SAT with visible degeneracy metadata.

    For initially separated, nondegenerate rectangles, P1 is the nonempty
    closest-ray set criterion in Eq. (8). Existing contact and zero relative
    motion are handled explicitly, outside the positive-ET derivation.
    """
    sat_time = swept_contact(a, b) if contact_time is _UNSET else contact_time
    if sat_time is not None and (not math.isfinite(sat_time) or sat_time < 0):
        raise ValueError('cached contact_time must be None or finite and nonnegative')
    if sat_time is not None and sat_time <= EPS:
        return SpatialInteraction(True, 0., method='initial_contact', initial_contact=True)

    a_c, b_c = closest_collision_sets(a, b)
    pairs = _minimum_distance_pairs(a_c + b_c)  # Eq. (9), including ties.
    if sat_time is None:
        if pairs:
            raise RuntimeError('ray P1 and SAT disagree; inspect numerical geometry')
        return SpatialInteraction(False, None, a_c, b_c)

    speed = hypot(*sub(a.velocity, b.velocity))
    if speed <= EPS:
        raise RuntimeError('positive contact time with zero relative speed')
    if pairs:
        paper_time = contact_time_from_pairs(pairs, speed)
        if math.isclose(paper_time, sat_time, rel_tol=CONTACT_TIME_RTOL, abs_tol=CONTACT_TIME_RTOL):
            return SpatialInteraction(True, sat_time, a_c, b_c, pairs)

    # Eq. (5) skips singular rays; this is explicitly labelled a completion.
    recovered = tuple(contact_pairs_at_time(a, b, sat_time))
    if not recovered:
        raise RuntimeError('could not recover contact pairs at SAT contact time')
    return SpatialInteraction(True, sat_time, a_c, b_c, recovered,
                              method='sat_degenerate_contact_completion')


def earliest_ab_pairs(a, b, contact_time=_UNSET):
    """Compatibility API returning (ET,list[V_i,E_i]); see spatial_interaction."""
    result = spatial_interaction(a, b, contact_time)
    if result.initial_contact:
        raise ValueError('A and B overlap or touch initially')
    return result.contact_time, list(result.pairs)


@njit(cache=True)
def _native_roots(events, count, c0, c1, c2, width):
    if abs(c2) < 1e-14:
        if abs(c1) >= 1e-14:
            r = -c0 / c1
            if -EPS <= r <= width + EPS:
                events[count] = max(0.0, min(width, r))
                count += 1
        return count
    disc = c1 * c1 - 4 * c2 * c0
    if disc < -1e-12:
        return count
    q = -0.5 * (c1 + math.copysign(math.sqrt(max(0.0, disc)), c1))
    if abs(q) > 1e-14:
        r1, r2 = (q / c2, c0 / q)
    else:
        r1 = -c1 / (2 * c2)
        r2 = r1
    if -EPS <= r1 <= width + EPS:
        events[count] = max(0.0, min(width, r1))
        count += 1
    if -EPS <= r2 <= width + EPS:
        events[count] = max(0.0, min(width, r2))
        count += 1
    return count


@njit(cache=True)
def _native_overlaps(polys, radii, t):
    for k in range(4):
        value = polys[k, 0] + polys[k, 1] * t + polys[k, 2] * t * t
        if abs(value) > radii[k] + EPS:
            return False
    return True


def predicted_ttc(a,b):
    """Exact constant-velocity TTC"""
    position=(b.x-a.x,b.y-a.y);velocity=(b.vx-a.vx,b.vy-a.vy)
    entry,exit=-math.inf,math.inf
    for axis in list(a.basis)+list(b.basis):
        r=a.radius(axis)+b.radius(axis)
        p=np.dot(position,axis);v=np.dot(velocity,axis)
        if abs(v)<EPS:
            if abs(p)>r+EPS:return None
            continue
        t1,t2=(-r-p)/v,(r-p)/v
        entry=max(entry,min(t1,t2));exit=min(exit,max(t1,t2))
        if entry>exit+EPS:return None
    return None if exit<-EPS else max(0.,entry)


@njit(cache=True)
def cv_window(a,b,axes,radii):
    entry=-math.inf;exit_=math.inf
    for k in range(4):
        p=axes[k,0]*(b[0]-a[0])+axes[k,1]*(b[1]-a[1])
        v=axes[k,0]*(b[2]-a[2])+axes[k,1]*(b[3]-a[3])
        if abs(v)<EPS:
            if abs(p)>radii[k]+EPS:return math.inf,-math.inf
            continue
        t1=(-radii[k]-p)/v;t2=(radii[k]-p)/v
        entry=max(entry,min(t1,t2));exit_=min(exit_,max(t1,t2))
        if entry>exit_+EPS:return math.inf,-math.inf
    if exit_<-EPS:return math.inf,-math.inf
    return max(0.,entry),exit_
