"""First geometric contact forecasts and delayed surrounding responses.

The sampled trigger uses P1 / constant-velocity rectangle geometry, not P2.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from .constants import EPS
from .motion import (_native_motion, _native_add_knots, _native_road, _native_zone)
from .geometry import _native_roots, _native_overlaps
from . import geometry

def sample_times(duration: float, interval: float) -> np.ndarray:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and positive')
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('sample interval must be finite and positive')
    values = np.arange(math.floor(duration / interval) + 1) * interval
    values = values[values < duration - EPS]
    return np.r_[values, duration]


@njit(cache=True)
def delayed_motion(state, acceleration, trigger, time, cfg):
    """Coast up to trigger; apply acceleration only to the remaining time."""
    if time <= trigger:
        return (state[1] + state[3]*time, state[2] + state[4]*time,
                state[3], state[4], 0., 0.)
    shifted = state.copy()
    shifted[1] += state[3]*trigger
    shifted[2] += state[4]*trigger
    return _native_motion(shifted, acceleration, 0., False,
                               time-trigger, cfg)


@njit(cache=True)
def delayed_contact(sv, other, axes, radii, ax, ay, response, trigger, end, cfg):
    """Exact contact time on the piecewise-quadratic delayed-response paths."""
    if end <= EPS:
        return math.inf
    local = cfg.copy()
    local[0] = end
    cuts = np.empty(12)
    cuts[0] = 0.
    cuts[1] = end
    n = _native_add_knots(cuts, 2, sv, ax, ay, True, local)
    if EPS < trigger < end-EPS:
        cuts[n] = trigger
        n += 1
    if response < 0.:
        stop = trigger + other[11]/(-response)
        if EPS < stop < end-EPS:
            cuts[n] = stop
            n += 1
    cuts = np.sort(cuts[:n])
    polys = np.empty((4, 3))
    events = np.empty(18)
    for segment in range(len(cuts)-1):
        left, right = cuts[segment], cuts[segment+1]
        width = right-left
        if width <= 0.:
            continue
        a = _native_motion(sv, ax, ay, True, left, cfg)
        b = delayed_motion(other, response, trigger, left, cfg)
        am = _native_motion(sv, ax, ay, True, (left+right)/2, cfg)
        bm = delayed_motion(other, response, trigger, (left+right)/2, cfg)
        separated = False
        for k in range(4):
            nx, ny = axes[k, 0], axes[k, 1]
            c0 = nx*(b[0]-a[0]) + ny*(b[1]-a[1])
            c1 = nx*(b[2]-a[2]) + ny*(b[3]-a[3])
            c2 = .5*(nx*(bm[4]-am[4]) + ny*(bm[5]-am[5]))
            polys[k, 0], polys[k, 1], polys[k, 2] = c0, c1, c2
            finish = c0+c1*width+c2*width*width
            low, high = min(c0, finish), max(c0, finish)
            if abs(c2) > 1e-14:
                stationary = -c1/(2*c2)
                if 0. < stationary < width:
                    value = c0+c1*stationary+c2*stationary*stationary
                    low, high = min(low, value), max(high, value)
            if low > radii[k]+EPS or high < -radii[k]-EPS:
                separated = True
                break
        if separated:
            continue
        events[0], events[1] = 0., width
        count = 2
        for k in range(4):
            count = _native_roots(events, count, polys[k, 0]+radii[k],
                                      polys[k, 1], polys[k, 2], width)
            count = _native_roots(events, count, polys[k, 0]-radii[k],
                                      polys[k, 1], polys[k, 2], width)
        ordered = np.sort(events[:count])
        for k in range(count):
            if _native_overlaps(polys, radii, ordered[k]):
                return left+ordered[k]
            if (k+1 < count and _native_overlaps(
                    polys, radii, (ordered[k]+ordered[k+1])/2)):
                return left+ordered[k]
    return math.inf


@njit(cache=True)
def first_trigger(sv_samples, other, times, axes, physical_radii):
    for k in range(len(times)):
        s = times[k]
        c = (other[1]+other[3]*s, other[2]+other[4]*s,
             other[3], other[4], 0., 0.)
        et, _ = geometry.cv_window(sv_samples[k], c, axes, physical_radii)
        if math.isfinite(et):
            return k, et
    return -1, math.inf


@njit(cache=True)
def response_static_safe(other, response, trigger, cfg, za, zl, zu, origin):
    """Check the entire delayed actor path, not just its response endpoint."""
    local = cfg.copy()
    if trigger > EPS:
        local[0] = trigger
        if not _native_road(other, 0., 0., False, local):
            return False
        if _native_zone(other, 0., 0., False, za, zl, zu, origin, local):
            return False
    remaining = cfg[0]-trigger
    if remaining > EPS:
        shifted = other.copy()
        shifted[1] += other[3]*trigger
        shifted[2] += other[4]*trigger
        local[0] = remaining
        if not _native_road(shifted, response, 0., False, local):
            return False
        if _native_zone(shifted, response, 0., False,
                             za, zl, zu, origin, local):
            return False
    return True


@njit(cache=True)
def analytic_response(sv, other, ax, ay, trigger, is_rear, cfg, terminal_clear):
    """Closed-form maximum of -2*G(u)/(direction_x*u^2), segment by segment.

    G is the longitudinal net gap without a surrounding response. A rear actor
    uses negative acceleration; a front actor uses positive acceleration. With
    a constant SV longitudinal acceleration this is exactly the original MLC
    max{0, max_u[-a_x-2*delta_v/u-2*D/u^2]} rear formula. The 1e-7-m guard
    prevents numerical contact with the existing expanded safety boundary.
    """
    remaining = cfg[0]-trigger
    direction_x = other[9]
    if remaining <= EPS or direction_x <= EPS:
        return math.nan, 0, math.nan, math.nan
    side = 1. if is_rear else -1.
    radius_sv = abs(sv[5])*sv[7]+abs(sv[6])*sv[8]
    radius_c = abs(other[5])*other[7]+abs(other[6])*other[8]
    clearance = radius_sv+radius_c+cfg[3]+1e-7
    initial_sv = _native_motion(sv, ax, ay, True, trigger, cfg)
    gap = side*(initial_sv[0]-(other[1]+other[3]*trigger))-clearance
    if gap <= 0.:
        return math.nan, 0, math.nan, math.nan
    cuts = np.empty(8)
    cuts[0] = trigger
    cuts[1] = cfg[0]
    raw = np.empty(8)
    nraw = _native_add_knots(raw, 0, sv, ax, ay, True, cfg)
    count = 2
    for k in range(nraw):
        if trigger+EPS < raw[k] < cfg[0]-EPS:
            cuts[count] = raw[k]
            count += 1
    cuts = np.sort(cuts[:count])
    required, evaluations, critical_time = 0., 0, remaining
    for k in range(len(cuts)-1):
        left, right = cuts[k], cuts[k+1]
        if right-left <= EPS:
            continue
        uleft, uright = left-trigger, right-trigger
        at = _native_motion(sv, ax, ay, True, left, cfg)
        middle = _native_motion(sv, ax, ay, True, (left+right)/2, cfg)
        gleft = side*(at[0]-(other[1]+other[3]*left))-clearance
        velocity = side*(at[2]-other[3])
        acceleration = side*middle[4]
        # G(u) = A + B*u + C*u^2 within this phase.
        A = gleft-velocity*uleft+.5*acceleration*uleft*uleft
        B = velocity-acceleration*uleft
        C = .5*acceleration
        candidates = np.empty(3)
        candidates[0], candidates[1] = uleft, uright
        size = 2
        if abs(B) > 1e-14:
            stationary = -2*A/B
            if uleft < stationary < uright:
                candidates[2] = stationary
                size = 3
        for j in range(size):
            u = candidates[j]
            if u <= 1e-12:
                continue
            value = -2*(A/u/u+B/u+C)/direction_x
            evaluations += 1
            if value > required:
                required, critical_time = value, u
    endpoint = _native_motion(sv, ax, ay, True, cfg[0], cfg)
    terminal_bound = 0.
    if terminal_clear:
        # Sufficient longitudinal ordering for no future CV contact after T.
        # This is an explicit extra test option, NOT an updated A-B ET.
        terminal_bound = max(0., side*(other[3]-endpoint[2])/(direction_x*remaining))
        required = max(required, terminal_bound)
    return (-required if is_rear else required), evaluations, critical_time, terminal_bound
