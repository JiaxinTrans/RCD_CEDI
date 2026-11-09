"""MLC CEDI with waiting time, InDepth/ET and lane-wise interaction sums.

Preserves the supplied MLC model: continuous original-front waiting clearance,
analytic rear demand, target-slot checks at initiation, first feasible stage.
This is not a full SAT simulation of the subsequent lane-change trajectory.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from ..constants import G
from ..models import VehicleState as State
from ..config import MLCConfig as Config

def lane_id(d: float, cfg: Config) -> int:
    return math.floor((d - cfg.lane_origin) / cfg.lane_width + 0.5)


def cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def longitudinal_grid(cfg: Config, stage: int, steps: int) -> np.ndarray:
    """Retain the initial grid points when the search bounds expand."""
    positive_steps = max(5, steps // 4)
    grids = [np.linspace(-cfg.ax_brake_limits_g[0] * G, 0.0, steps),
             np.linspace(0.0, cfg.ax_accel_limits_g[0] * G, positive_steps)]
    if stage == 1:
        grids.extend((np.linspace(-cfg.ax_brake_limits_g[1] * G, 0.0, steps),
                      np.linspace(0.0, cfg.ax_accel_limits_g[1] * G,
                                  positive_steps)))
    return np.unique(np.concatenate(grids))


@lru_cache(maxsize=8192)
def corner_offsets(state: State) -> tuple[tuple[float, float], ...]:
    c, sn = math.cos(state.theta), math.sin(state.theta)
    return tuple((along * state.length * c / 2 - across * state.width * sn / 2,
                  along * state.length * sn / 2 + across * state.width * c / 2)
                 for along, across in ((-1, 1), (1, 1), (1, -1), (-1, -1)))


def zone_contact(state: State, s: float, d: float, cfg: Config) -> bool:
    # Finite segment: the corner projection must overlap its s range.
    offsets = corner_offsets(state)
    if s + max(p[0] for p in offsets) < cfg.zone_start[0]:
        return False
    if s + min(p[0] for p in offsets) > cfg.zone_end[0]:
        return False
    slope = (cfg.zone_end[1] - cfg.zone_start[1]) / (
        cfg.zone_end[0] - cfg.zone_start[0])
    side = 1 if cfg.target_lane_offset > 0 else -1
    return min(side * (d + dd - slope * (s + ds - cfg.zone_start[0]))
               for ds, dd in offsets) <= 1e-9


def first_zone_contact(state: State, ax: float, delay: float,
                       cfg: Config) -> float:
    """First corner intersection with a finite cone line after delay."""
    s = state.s + state.vs * delay + ax * delay * delay / 2
    d = state.d + state.vd * delay
    vs = state.vs + ax * delay
    if zone_contact(state, s, d, cfg):
        return 0.0
    if vs <= 0:
        return math.inf
    slope = (cfg.zone_end[1] - cfg.zone_start[1]) / (
        cfg.zone_end[0] - cfg.zone_start[0])
    roots: list[float] = []
    for ds, dd in corner_offsets(state):
        xs, yd = s + ds, d + dd
        c0 = yd - slope * (xs - cfg.zone_start[0])
        c1 = state.vd - slope * vs
        c2 = -0.5 * slope * ax
        if abs(c2) < 1e-13:
            values = [-c0 / c1] if abs(c1) > 1e-13 else []
        else:
            discriminant = c1 * c1 - 4 * c2 * c0
            values = [] if discriminant < 0 else [
                (-c1 - math.sqrt(discriminant)) / (2 * c2),
                (-c1 + math.sqrt(discriminant)) / (2 * c2)]
        for t in values:
            if t < -1e-10 or (ax < 0 and t > vs / -ax + 1e-10):
                continue
            t = max(0.0, t)
            hit_s = xs + vs * t + 0.5 * ax * t * t
            if cfg.zone_start[0] - 1e-8 <= hit_s <= cfg.zone_end[0] + 1e-8:
                roots.append(t)
    return min(roots, default=math.inf)


def indepth(state: State, s: float, d: float, vs: float,
            cfg: Config) -> float:
    length = math.hypot(vs, state.vd)
    if length <= 1e-12:
        return 0.0
    unit = (vs / length, state.vd / length)
    radius = max(abs(cross(p, unit)) for p in corner_offsets(state))
    half = ((cfg.zone_end[0] - cfg.zone_start[0]) / 2,
            (cfg.zone_end[1] - cfg.zone_start[1]) / 2)
    midpoint = ((cfg.zone_start[0] + cfg.zone_end[0]) / 2 - s,
                (cfg.zone_start[1] + cfg.zone_end[1]) / 2 - d)
    return max(0.0, abs(cross(unit, half)) + radius - abs(cross(midpoint, unit)))


def bumper_extent(state: State) -> float:
    return (abs(math.cos(state.theta)) * state.length
            + abs(math.sin(state.theta)) * state.width) / 2


def lateral_extent(state: State) -> float:
    return (abs(math.sin(state.theta)) * state.length
            + abs(math.cos(state.theta)) * state.width) / 2


def min_front_gap(ego: State, front: State, ax: float,
                  tau: float, cfg: Config) -> float:
    """Minimum longitudinal bumper gap until steering begins."""
    initial = front.s - ego.s - bumper_extent(front) - bumper_extent(ego)
    def gap(t: float) -> float:
        return initial + (front.vs - ego.vs) * t - 0.5 * ax * t * t
    times = [0.0, tau]
    if ax < 0:
        critical = (front.vs - ego.vs) / ax
        if 0 < critical < tau:
            times.append(critical)
    return min(gap(t) for t in times)


def _nearest_original(ego: State, others: list[State], cfg: Config
                       ) -> tuple[State | None, State | None]:
    same = [v for v in others if lane_id(v.d, cfg) == lane_id(ego.d, cfg)]
    front = min((v for v in same if v.s > ego.s),
                key=lambda v: (v.s-ego.s, v.vid), default=None)
    rear = min((v for v in same if v.s < ego.s),
               key=lambda v: (ego.s-v.s, v.vid), default=None)
    return front, rear


def _instant_demand(ego_s: float, ego_v: float, ego: State,
                    other: State | None, tau: float, following: bool
                    ) -> tuple[float, float]:
    if other is None:
        return 0.0, math.inf
    other_s = other.s+other.vs*tau
    distance = (ego_s-other_s if following else other_s-ego_s)
    clearance = distance-bumper_extent(ego)-bumper_extent(other)
    if clearance <= 0.0:
        return math.inf, clearance
    closing = max(0.0, other.vs-ego_v if following else ego_v-other.vs)
    return closing*closing/(2.0*clearance), clearance


def mlc_rear_demand(ego: State, rear: State | None, ax: float,
                    tau: float, cfg: Config) -> float:
    """Analytic manuscript demand, retaining the configured clearance margin.

    With forward-moving SV, clipping a follower's motion at rest cannot
    introduce a later collision: clearance increases after it stops.
    """
    if rear is None:
        return 0.0
    clearance = ego.s - rear.s - bumper_extent(ego) - bumper_extent(rear)
    available = clearance - cfg.gap
    if available < 0.0:
        return math.inf
    if tau == 0.0:
        return 0.0
    dv = ego.vs - rear.vs
    if available == 0.0 and dv < 0.0:
        return math.inf
    times = [tau]
    if dv < 0.0 and available > 0.0:
        critical = -2.0 * available / dv
        if critical < tau:
            times.append(critical)
    return max(0.0, *(-ax - 2.0*dv/t - 2.0*available/t**2 for t in times))


def evaluate_mlc_candidate(ego: State, others: list[State], ax: float,
                 tau: float, cfg: Config, before: float,
                 stage: int) -> dict | None:
    """Nearest-neighbor MLC cost and lane-gap constraints; no SAT sweep."""
    if (tau < 0.0 or before <= tau + 1e-8
            or ego.vs + ax*tau <= 0.0 or (tau == 0.0 and ax != 0.0)):
        return None
    s = ego.s + ego.vs*tau + 0.5*ax*tau*tau
    d = ego.d + ego.vd*tau
    vs = ego.vs + ax*tau
    target_lane = lane_id(ego.d, cfg) + cfg.target_lane_offset
    target_center = cfg.lane_origin + target_lane*cfg.lane_width
    extent = lateral_extent(ego)
    if (min(ego.d, d)-extent < cfg.road_min_d
            or max(ego.d, d)+extent > cfg.road_max_d
            or target_center-extent < cfg.road_min_d
            or target_center+extent > cfg.road_max_d):
        return None

    original_front, original_rear = _nearest_original(ego, others, cfg)
    # A same-position original-lane car must not disappear from f/r selection.
    if any(lane_id(v.d, cfg) == lane_id(ego.d, cfg) and v.s == ego.s
           for v in others):
        return None
    if (original_front is not None
            and min_front_gap(ego, original_front, ax, tau, cfg) < cfg.gap):
        return None
    b_or = mlc_rear_demand(ego, original_rear, ax, tau, cfg)
    if not math.isfinite(b_or):
        return None

    target = [v for v in others if abs(v.d+v.vd*tau-target_center)
              <= extent+lateral_extent(v)]
    # Check the whole candidate slot at initiation, including side-by-side
    # cars; only its nearest bounding cars contribute to the cost.
    if any(abs(v.s+v.vs*tau-s)-bumper_extent(ego)-bumper_extent(v)
           < cfg.gap for v in target):
        return None
    front = min((v for v in target if v.s+v.vs*tau > s),
                key=lambda v: (v.s+v.vs*tau-s, v.vid), default=None)
    rear = min((v for v in target if v.s+v.vs*tau < s),
               key=lambda v: (s-v.s-v.vs*tau, v.vid), default=None)
    b_tf, d_tf = _instant_demand(s, vs, ego, front, tau, False)
    b_tr, d_tr = _instant_demand(s, vs, ego, rear, tau, True)
    b_of, d_of = _instant_demand(s, vs, ego, original_front, tau, False)
    if not all(math.isfinite(b) for b in (b_of, b_tf, b_tr)):
        return None
    start = State(ego.vid, s, d, vs, ego.vd, ego.theta, ego.length, ego.width)
    et = first_zone_contact(start, 0.0, 0.0, cfg)
    if not math.isfinite(et) or et <= 1e-9:
        return None
    depth = indepth(ego, s, d, vs, cfg)
    if depth <= 1e-9:
        return None
    ay = 2.0*depth/(et*et)
    a_front, a_rear = b_of+b_tf, b_or+b_tr
    reference_x = (cfg.ax_accel_limits_g[0] if ax > 0.0
                   else cfg.ax_brake_limits_g[0])*G
    interaction = math.hypot(a_front/(cfg.ax_brake_limits_g[0]*G),
                             a_rear/(cfg.surrounding_brake_limits_g[0]*G))
    score = math.hypot(ax/reference_x, ay/(cfg.ay_limits_g[0]*G), interaction)
    if not math.isfinite(score):
        return None
    # Return in-memory demand mappings and ID tuples, without serialization.
    fd, rd = {}, {}
    for car, b in ((original_front, b_of), (front, b_tf)):
        if car is not None:
            fd[car.vid] = fd.get(car.vid, 0.0) + b
    for car, b in ((original_rear, b_or), (rear, b_tr)):
        if car is not None:
            rd[car.vid] = rd.get(car.vid, 0.0) + b
    ids = lambda cars: tuple(v.vid for v in cars if v is not None)
    return {'CEDI': score, 'ax_mps2': ax, 'ay_mps2': ay,
            'a_s_mps2': a_rear, 'a_s_max_mps2': max(b_or, b_tr),
            'a_rear_mps2': b_or, 'a_front_mps2': a_front,
            'a_front_max_mps2': max(b_of, b_tf),
            'a_target_mps2': max(b_tf, b_tr),
            'interaction_cost_norm': interaction,
            'follower_demands_mps2': rd,
            'front_demands_mps2': fd,
            'tau_s': tau, 'ET_s': et, 'InDepth_m': depth,
            'target_lane_id': target_lane,
            'target_vehicle_ids': ids((front, rear)),
            'rear_vehicle_ids': ids((original_rear,)),
            'front_same_vehicle_ids': ids((original_front,)),
            'ahead_of_target_ids': ids((front,)),
            'behind_target_ids': ids((rear,)),
            'critical_surrounding_id': max(rd, key=rd.get) if any(rd.values()) else 0,
            'critical_front_id': max(fd, key=fd.get) if any(fd.values()) else 0}


def solve_mlc(ego: State, others: list[State], cfg: Config) -> dict:
    contact = first_zone_contact(ego, 0.0, 0.0, cfg)
    if contact == 0.0:
        return {'status': 'already_at_work_zone'}
    if not math.isfinite(contact):
        return {'status': 'no_predicted_work_zone_contact'}
    initial_times = np.linspace(0.0, min(cfg.horizon, contact), cfg.tau_steps)
    # The search interval cannot extend beyond the initially available ET.
    expanded_times = initial_times
    # Reuse the existing search-resolution setting as a finite stage budget.
    # Budget exhaustion is not proof of physical infeasibility.
    total_evaluations = 0
    previous_grid = np.array([], dtype=float)
    for stage in range(max(2, cfg.ax_steps)):
        times = initial_times if stage == 0 else expanded_times
        if stage < 2:
            ax_grid = longitudinal_grid(cfg, stage, cfg.ax_steps)
        else:
            factor = 2.0**(stage-1)
            ax_grid = np.unique(np.r_[previous_grid,
                np.linspace(-cfg.ax_brake_limits_g[1]*G*factor, 0.0,
                            cfg.ax_steps),
                np.linspace(0.0, cfg.ax_accel_limits_g[1]*G*factor,
                            max(5, cfg.ax_steps//4))])
        # Old grid points were already tested with expanded times at stage 1.
        active_grid = (ax_grid if stage < 2 else
                       np.setdiff1d(ax_grid, previous_grid))
        previous_grid = ax_grid
        before_by_ax = {float(ax): first_zone_contact(ego, float(ax), 0.0, cfg)
                        for ax in active_grid}
        best = None
        for tau in times:
            tau = float(tau)
            for ax in active_grid:
                ax = float(ax)
                ref = (cfg.ax_accel_limits_g[0] if ax > 0.0
                       else cfg.ax_brake_limits_g[0])*G
                if best is not None and abs(ax)/ref >= best['CEDI']:
                    continue
                if tau == 0.0 and ax != 0.0:
                    continue
                total_evaluations += 1
                candidate = evaluate_mlc_candidate(ego, others, ax, tau, cfg,
                                         before_by_ax[ax], stage)
                if candidate and (best is None or candidate['CEDI'] < best['CEDI']):
                    best = candidate
        if best is not None:
            return {'status': 'feasible',
                    'search_stage': 'initial' if stage == 0 else 'expanded',
                    **best}
    # This status it means no action was found in
    # the finite numerical search, not proof of continuous infeasibility.
    return {'status': 'no_feasible_action', 'search_stage': 'expanded'}
