"""Vehicle CEDI: fixed initial ET and minimum over final nested action set.

Rear vehicles execute delayed braking. Front future terms quantify subsequent
SV braking demand; they do not execute a second plan or replan the initial ET.
"""
from __future__ import annotations
import math
from time import perf_counter
import numpy as np
from numba import njit
from ..constants import EPS, G
from ..models import VehicleState as State
from ..geometry import predicted_ttc
from ..static import pack_scene, initial_zone_overlap
from ..motion import (_native_motion, _native_contact, _native_road, _native_zone)
from .. import geometry, responses
from .. import rcd as raw_rcd
from .. import audit as audit_tools
from ..search import candidate_grids

COUNTERS = ('candidates', 'SV_road', 'SV_zone', 'AB_contact', 'terminal_AB',
            'surrounding_queries', 'actual_contacts_before_trigger',
            'unsupported_gap_geometry', 'rear_response_limit',
            'rear_response_static', 'surrounding_contact', 'front_contact',
            'future_braking_limit', 'cost_pruned', 'feasible_candidates',
            'own_bound_stop')

def prioritize_packed_scene(packed, gap):
    """Order surrounding checks by the initial geometric RCD contact time.

    This is only an evaluation order. No actor or SV candidate is removed.
    The original interacting pair remains at indices 0 and 1. All associated
    geometric arrays are permuted together, retaining continuous zone checks.
    """
    actors, states, axes, radii, za, zl, zu, origin = packed
    if len(states) <= 3:
        return packed
    a = (states[0,1], states[0,2], states[0,3], states[0,4], 0., 0.)
    priorities = []
    for j in range(2, len(states)):
        b = (states[j,1], states[j,2], states[j,3], states[j,4], 0., 0.)
        et, _ = geometry.cv_window(a, b, axes[0,j], radii[0,j]-gap)
        priorities.append((float(et), j))
    order = np.array([0, 1]+[j for _, j in sorted(priorities)], dtype=int)
    if np.array_equal(order, np.arange(len(states))):
        return packed
    return ([actors[j] for j in order], np.ascontiguousarray(states[order]),
        np.ascontiguousarray(axes[order[:,None], order[None,:]]),
        np.ascontiguousarray(radii[order[:,None], order[None,:]]),
        np.ascontiguousarray(za[order]), np.ascontiguousarray(zl[order]),
        np.ascontiguousarray(zu[order]), origin)


@njit(cache=True)
def x_clearance(sv, other, cfg):
    return (abs(sv[5])*sv[7] + abs(sv[6])*sv[8]
            + abs(other[5])*other[7] + abs(other[6])*other[8]
            + cfg[3] + 1e-7)


@njit(cache=True)
def linear_tail_bound(gap_end, gap_velocity, remaining, direction_x):
    """sup[-2(A+B*r)/(k*r**2)] over r>=remaining, without time marching.

G(r)=gap_end+gap_velocity*(r-remaining). The response starts at r=0,
not at the SV endpoint; therefore this is not endpoint DRAC pasted onto an
earlier response. Negative vehicle speed is clamped by the motion evaluator.
"""
    if direction_x <= EPS or (remaining <= EPS and gap_end <= 0.):
        return math.nan, math.nan
    a = gap_end-gap_velocity*remaining
    best, at = 0., math.inf
    if remaining > EPS:
        value = -2.*gap_end/(direction_x*remaining*remaining)
        if value > best:
            best, at = value, remaining
    if abs(gap_velocity) > 1e-14:
        stationary = -2.*a/gap_velocity
        if stationary > EPS and stationary >= remaining-EPS:
            value = -2.*(a+gap_velocity*stationary)/(direction_x*stationary**2)
            if value > best:
                best, at = value, stationary
    return best, at


@njit(cache=True)
def front_tail_bound(sv_end, other_end, clearance):
    """SV brakes along its terminal velocity; C keeps its velocity.

Longitudinal ordering is a sufficient condition, not minimum 2-D steering
or braking. Used only after terminal geometric RCD confirms relevance.
"""
    gap = other_end[0]-sv_end[0]-clearance
    speed = math.hypot(sv_end[2], sv_end[3])
    if gap <= 0. or speed <= EPS:
        return math.nan, gap
    direction_x = sv_end[2]/speed
    closing = sv_end[2]-other_end[2]
    if direction_x <= EPS or closing <= 0.:
        return math.nan, gap
    return closing*closing/(2.*gap*direction_x), gap


@njit(cache=True)
def response_tail_contact(sv_end, other, acc, trigger, cfg, axes, radii):
    """Exact SV-C contact AFTER T, with C's actual continued braking.

    The SV coasts with its terminal velocity. C keeps the response started at
    its first trigger until stopped. No C-D interactions or ET replanning.
    """
    c_end = responses.delayed_motion(other, acc, trigger, cfg[0], cfg)
    sv = other.copy()
    sv[1:5] = sv_end[:4]
    sv[11] = math.hypot(sv_end[2], sv_end[3])
    if sv[11] > EPS:
        sv[9], sv[10] = sv_end[2]/sv[11], sv_end[3]/sv[11]
    c = other.copy()
    c[1:5] = c_end[:4]
    c[11] = math.hypot(c_end[2], c_end[3])
    if c[11] > EPS:
        c[9], c[10] = c_end[2]/c[11], c_end[3]/c[11]
    if acc >= -EPS or c[11] <= EPS:
        return geometry.cv_window(sv_end, c_end, axes, radii)[0]
    stop = c[11]/(-acc)
    local = cfg.copy()
    local[0] = stop
    hit = responses.delayed_contact(sv, c, axes, radii, 0., 0., acc,
                                    0., stop, local)
    if math.isfinite(hit):
        return hit
    a_stop = _native_motion(sv, 0., 0., False, stop, local)
    c_stop = _native_motion(c, acc, 0., False, stop, local)
    entry, _ = geometry.cv_window(a_stop, c_stop, axes, radii)
    return stop+entry if math.isfinite(entry) else math.inf


@njit(cache=True)
def check_action(states, axes, radii, cfg, za, zl, zu, origin,
                 times, ax, ay, include_tail, upper_bound):
    """Evaluate every surrounding vehicle with exact original-pair checks.

Return failure code, cost and per-C numeric diagnostic rows:
trigger, forecast, C acceleration, within-demand, tail-demand, demand,
endpoint CV TTC, tail-critical r, role, trigger-gap, endpoint-gap, finite-hit.
"""
    n = len(states)
    details = np.full((n, 12), math.nan)
    lower = math.hypot(ax/(cfg[5] if ax > 0. else cfg[6]), ay/cfg[7])
    if lower >= upper_bound:
        return 13, math.inf, details, 0
    if not _native_road(states[0], ax, ay, True, cfg):
        return 1, math.inf, details, 0
    if _native_zone(states[0], ax, ay, True, za[0], zl[0], zu[0], origin, cfg):
        return 2, math.inf, details, 0
    if _native_contact(states, axes, radii, 0, 1, ax, ay, True,
                            0., 0., False, cfg):
        return 3, math.inf, details, 0
    end_sv = _native_motion(states[0], ax, ay, True, cfg[0], cfg)
    end_ab = _native_motion(states[1], 0., 0., False, cfg[0], cfg)
    ab_ttc, _ = geometry.cv_window(end_sv, end_ab, axes[0, 1], radii[0, 1])
    if math.isfinite(ab_ttc):
        return 4, math.inf, details, 0
    samples = np.empty((len(times), 6))
    for k in range(len(times)):
        samples[k] = _native_motion(states[0], ax, ay, True, times[k], cfg)
    # Sum raw nonnegative demands by role BEFORE normalizing and squaring.
    # Individual rear demands still define individual braking trajectories.
    front_sum, rear_sum, queries = 0., 0., 0
    for j in range(2, n):
        point, forecast = responses.first_trigger(samples, states[j], times,
                                    axes[0, j], radii[0, j]-cfg[3])
        queries += point+1 if point >= 0 else len(times)
        hit = responses.delayed_contact(states[0], states[j], axes[0, j],
                radii[0, j], ax, ay, 0., cfg[0], cfg[0], cfg)
        coast_end = (states[j, 1]+states[j, 3]*cfg[0],
                     states[j, 2]+states[j, 4]*cfg[0],
                     states[j, 3], states[j, 4], 0., 0.)
        future_ttc, _ = geometry.cv_window(end_sv, coast_end,
                                           axes[0, j], radii[0, j]-cfg[3])
        details[j, 2:6] = 0.
        details[j, 6] = future_ttc
        details[j, 11] = hit
        if point < 0:
            # An actual between-sample contact must never be silently accepted.
            if math.isfinite(hit):
                return 6, math.inf, details, queries
            # The exact endpoint is a sampled state; positive physical tail
            # contact would contradict a missing first sampled trigger.
            if include_tail and math.isfinite(future_ttc):
                return 7, math.inf, details, queries
            continue
        trigger = times[point]
        remaining = cfg[0]-trigger
        details[j, 0], details[j, 1] = trigger, forecast
        if math.isfinite(hit) and hit <= trigger+EPS:
            return 6, math.inf, details, queries
        # A transient instantaneous CV prediction that the *candidate* itself
        # resolves needs no response. ET-after conflicts are not removed here.
        needs_in = math.isfinite(hit)
        needs_tail = include_tail and math.isfinite(future_ttc)
        clearance = x_clearance(states[0], states[j], cfg)
        at = samples[point]
        x_c = states[j, 1]+states[j, 3]*trigger
        is_rear = x_c < at[0]
        details[j, 8] = -1. if is_rear else 1.
        details[j, 9] = abs(at[0]-x_c)-clearance
        if not needs_in and not needs_tail:
            continue
        in_demand, tail_demand, c_acc = 0., 0., 0.
        if is_rear:
            if needs_in:
                c_acc, _, _, _ = responses.analytic_response(states[0], states[j],
                                    ax, ay, trigger, True, cfg, False)
                if not math.isfinite(c_acc):
                    return 7, math.inf, details, queries
                in_demand = -c_acc
            if needs_tail:
                gap_end = end_sv[0]-coast_end[0]-clearance
                details[j, 10] = gap_end
                tail_demand, critical = linear_tail_bound(gap_end,
                    end_sv[2]-coast_end[2], remaining, states[j, 9])
                details[j, 7] = critical
                # Do not invent an infinite demand at longitudinal-overlap
                # geometries; explicitly reject unsupported following formulas.
                if not math.isfinite(tail_demand) or details[j, 9] <= 0.:
                    return 7, math.inf, details, queries
            demand = max(in_demand, tail_demand)
            # The no-response terminal forecast is insufficient: braking C
            # changes its longitudinal AND lateral velocities. Audit that
            # actual response, even when needs_tail was initially false.
            if (include_tail and demand > EPS and
                    math.isfinite(response_tail_contact(end_sv, states[j],
                        -demand, trigger, cfg, axes[0,j], radii[0,j]))):
                gap_end = end_sv[0]-coast_end[0]-clearance
                details[j,10] = gap_end
                extra, critical = linear_tail_bound(gap_end,
                    end_sv[2]-coast_end[2], remaining, states[j,9])
                if not math.isfinite(extra) or details[j,9] <= 0.:
                    return 7, math.inf, details, queries
                tail_demand = max(tail_demand, extra)
                details[j,7] = critical
                demand = max(in_demand, tail_demand)
            c_acc = -demand
            if demand > cfg[11]+EPS:
                return 8, math.inf, details, queries
            # Only nonzero, SV-related responses receive C static constraints.
            if demand > EPS and not responses.response_static_safe(
                    states[j], c_acc, trigger, cfg, za[j], zl[j], zu[j], origin):
                return 9, math.inf, details, queries
            if (include_tail and demand > EPS and
                    math.isfinite(response_tail_contact(end_sv, states[j],
                        c_acc, trigger, cfg, axes[0,j], radii[0,j]))):
                return 10, math.inf, details, queries
        else:
            # A preceding C never accelerates to rescue an unsafe SV action.
            # Within-ET contact must be removed by a different SV candidate.
            if needs_in:
                return 11, math.inf, details, queries
            if needs_tail:
                tail_demand, gap_end = front_tail_bound(end_sv, coast_end, clearance)
                details[j, 10] = gap_end
                if not math.isfinite(tail_demand):
                    return 7, math.inf, details, queries
                if tail_demand > cfg[12]+EPS:
                    return 12, math.inf, details, queries
            demand = tail_demand
        if is_rear:
            rear_sum += demand
        else:
            front_sum += demand
        environment_squared = (front_sum/cfg[6])**2 + (rear_sum/cfg[8])**2
        if math.sqrt(lower*lower + environment_squared) >= upper_bound:
            return 13, math.inf, details, queries
        if math.isfinite(responses.delayed_contact(states[0], states[j], axes[0, j],
                radii[0, j], ax, ay, c_acc, trigger, cfg[0], cfg)):
            return 10, math.inf, details, queries
        details[j, 2] = c_acc
        details[j, 3] = in_demand
        details[j, 4] = tail_demand
        details[j, 5] = demand
    environment_squared = (front_sum/cfg[6])**2 + (rear_sum/cfg[8])**2
    return 0, math.sqrt(lower*lower + environment_squared), details, queries


@njit(cache=True, nogil=True)
def search(states, axes, radii, candidates, cfg, za, zl, zu, origin, times, include_tail):
    best, best_ax, best_ay, best_index = math.inf, math.nan, math.nan, -1
    best_details = np.full((len(states), 12), math.nan)
    counts = np.zeros(len(COUNTERS), np.int64)
    for index in range(len(candidates)):
        lower, ax, ay = candidates[index]
        if lower >= best:
            counts[15] += 1
            break
        counts[0] += 1
        code, cost, details, queries = check_action(states, axes, radii, cfg,
                za, zl, zu, origin, times, ax, ay, include_tail, best)
        counts[5] += queries
        if code:
            counts[code] += 1
            continue
        counts[14] += 1
        if cost < best:
            best, best_ax, best_ay, best_index = cost, ax, ay, index
            best_details[:] = details
    return best, best_ax, best_ay, best_index, best_details, counts


def solve(ego, opponent, scene, polygon, settings, grids, factors,
          sample_s=.1, include_tail=True, expand_response_limits=True, audit=True):
    start = perf_counter()
    et = predicted_ttc(ego, opponent)
    result = dict(subject_vehicle=ego.vid, opponent_vehicle=opponent.vid,
        status='no_initial_AB_conflict', CEDI=0., ax_mps2=None, ay_mps2=None,
        original_ET_s=et, planning_duration_s=None, expansion_factor=1.,
        expansion_level=0, responses=[], attempts=[],
        tail_included=include_tail, normalization_changed=False,
        terminal_AB_RCD_conflicts=None, candidates_visited=0)
    if et is None or et <= EPS or initial_zone_overlap(ego, polygon, settings.gap):
        if et is not None:
            result.update(status='initial_AB_contact' if et <= EPS
                          else 'initial_SV_work_zone_overlap', CEDI=None)
        result['elapsed_ms'] = (perf_counter()-start)*1000
        return result
    initial_radius = ego.radius((0.,1.))+settings.gap
    if (ego.y-initial_radius < settings.road_min-EPS or
            ego.y+initial_radius > settings.road_max+EPS):
        result.update(status='initial_SV_road_boundary_violation',CEDI=None)
        result['elapsed_ms'] = (perf_counter()-start)*1000
        return result
    duration = et  # Initial ET is fixed; plan_cap never truncates it.
    result['planning_duration_s'] = duration
    packed = prioritize_packed_scene(
        pack_scene(ego, opponent, scene, polygon, settings), settings.gap)
    actors, states, axes, radii, za, zl, zu, origin = packed
    times = responses.sample_times(duration, sample_s)
    cfg = np.r_[settings.numeric(), settings.brake_step, 0.,
                settings.surround_brake_g*G, settings.ax_brake_g*G]
    cfg[0] = duration
    if not factors or len(grids) != len(factors):
        raise ValueError('One nested candidate grid is required per expansion factor')
    # The final nested grid contains all reported candidates. Use one common
    # admissible-response set, then optimize it globally in lower-bound order.
    # Early stopping at the first feasible expansion would not minimize this set.
    level = len(factors)-1
    factor, grid = factors[-1], grids[-1]
    cfg[11] = settings.surround_brake_g*G*(factor if expand_response_limits else 1.)
    cfg[12] = settings.ax_brake_g*G*factor
    t0 = perf_counter()
    best, ax, ay, index, details, counts = search(states, axes, radii,
                grid, cfg, za, zl, zu, origin, times, include_tail)
    result['attempts'].append(dict(expansion_level=level, factor=factor,
        n_candidates=len(grid), elapsed_ms=(perf_counter()-t0)*1000,
        feasible=math.isfinite(best), counters=dict(zip(COUNTERS, map(int, counts))),
        search_rule='minimum over the final nested candidate set'))
    result['candidates_visited'] = int(counts[0])
    if math.isfinite(best):
        # Existing expansion columns describe the smallest configured envelope
        # admitting the retained control and each individual response.
        rear_max = max((details[j,5] for j in range(2,len(states))
                        if details[j,8] == -1.), default=0.)
        front_max = max((details[j,5] for j in range(2,len(states))
                         if details[j,8] == 1.), default=0.)
        for k, (f, g) in enumerate(zip(factors, grids)):
            present = np.any((g[:,1] == ax) & (g[:,2] == ay))
            rear_cap = settings.surround_brake_g*G*(f if expand_response_limits else 1.)
            front_cap = settings.ax_brake_g*G*f
            if present and rear_max <= rear_cap+EPS and front_max <= front_cap+EPS:
                level, factor = k, f
                break
        result.update(status='feasible', CEDI=float(best), ax_mps2=float(ax),
            ay_mps2=float(ay), expansion_factor=float(factor), expansion_level=level)
    if not math.isfinite(best):
        result.update(status='no_feasible_action_in_expanded_search', CEDI=None,
            expansion_level=len(factors)-1, expansion_factor=float(factors[-1]))
    else:
        end_sv = _native_motion(states[0], ax, ay, True, duration, cfg)
        if audit:
            # Re-evaluate WITHOUT cost pruning and audit original RCD calls.
            code, cost, checked, _ = check_action(states, axes, radii, cfg, za,
                    zl, zu, origin, times, ax, ay, include_tail, math.inf)
            if code or not math.isclose(cost, best, abs_tol=1e-11):
                raise AssertionError(('retained candidate audit', code, cost, best))
            if not np.allclose(checked, details, equal_nan=True, atol=1e-11, rtol=0.):
                raise AssertionError('retained diagnostic mismatch')
            end_other = _native_motion(states[1], 0., 0., False, duration, cfg)
            a = State(ego.vid, *end_sv[:4], ego.heading, ego.length, ego.width)
            b = State(opponent.vid, *end_other[:4], opponent.heading, opponent.length, opponent.width)
            if predicted_ttc(a, b) is not None:
                raise AssertionError('terminal original pair contact')
            end_states = states.copy()
            for i, at in ((0, end_sv), (1, end_other)):
                end_states[i, 1:5] = at[:4]
            original_audit = audit_tools.raw_rcd_audit(end_states,
                                np.zeros(len(states), dtype=np.bool_), pair_only=True)
            if original_audit['geometric_contacts']:
                raise AssertionError('original RCD terminal audit failed')
            for j in range(2, len(states)):
                if not math.isfinite(details[j, 0]):
                    continue
                trigger = details[j, 0]
                at = _native_motion(states[0], ax, ay, True, trigger, cfg)
                c = states[j]
                first = raw_rcd.Vehicle('SV', (at[0],at[1]), (at[2],at[3]),
                    2*states[0,7], 2*states[0,8], math.degrees(ego.heading))
                second = raw_rcd.Vehicle(str(actors[j].vid),
                    (c[1]+c[3]*trigger, c[2]+c[4]*trigger), (c[3],c[4]),
                    2*c[7], 2*c[8], math.degrees(actors[j].heading))
                raw_time = raw_rcd.swept_contact(first, second)
                if raw_time is None or not math.isclose(raw_time, details[j,1], abs_tol=1e-7, rel_tol=1e-7):
                    raise AssertionError(('first RCD trigger audit', raw_time, details[j,1]))
            result['terminal_AB_RCD_conflicts'] = 0
            result['retained_action_audit'] = 'passed'
        for j in range(2, len(actors)):
            row = details[j]
            role = 'rear' if row[8] == -1. else 'front' if row[8] == 1. else 'inactive'
            result['responses'].append(dict(vehicle_id=actors[j].vid, role=role,
                first_trigger_s=float(row[0]) if math.isfinite(row[0]) else None,
                trigger_forecast_TTC_s=float(row[1]) if math.isfinite(row[1]) else None,
                C_response_ax_mps2=float(row[2]), within_ET_demand_mps2=float(row[3]),
                future_demand_mps2=float(row[4]), total_demand_mps2=float(row[5]),
                terminal_coast_TTC_s=float(row[6]) if math.isfinite(row[6]) else None,
                tail_critical_elapsed_s=float(row[7]) if math.isfinite(row[7]) else None,
                net_gap_at_trigger_m=float(row[9]) if math.isfinite(row[9]) else None,
                net_gap_at_end_m=float(row[10]) if math.isfinite(row[10]) else None,
                within_ET_coast_contact_s=float(row[11]) if math.isfinite(row[11]) else None,
                closing_speed_at_end_mps=(float(states[j,3]-end_sv[2]) if role == 'rear'
                    else float(end_sv[2]-states[j,3]) if role == 'front' else None),
                terminal_SV_direction_x=float(end_sv[2]/math.hypot(end_sv[2],end_sv[3]))
                    if math.hypot(end_sv[2],end_sv[3]) > EPS else None,
                future_response_start_s=float(row[0]) if role == 'rear' and math.isfinite(row[0])
                    else duration if role == 'front' else None,
                demand_bearer='following_C' if role == 'rear' else 'SV_after_action' if role == 'front' else 'none',
                normalized_demand=float(row[5])/(settings.surround_brake_g*G
                    if role == 'rear' else settings.ax_brake_g*G)
                    if role != 'inactive' else 0.))
    result['elapsed_ms'] = (perf_counter()-start)*1000
    return result
