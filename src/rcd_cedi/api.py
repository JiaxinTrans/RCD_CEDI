"""In-memory APIs: explicit vehicle perspectives, no file I/O or event filters."""
from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache

from .config import MLCConfig, VehicleConfig
from .constants import EPS
from .static import StaticEnvironment
from .geometry import Vehicle, earliest_ab_pairs, swept_contact
from .models import CEDIResult, RCDResult, VehicleState
from . import rcd
from .cedi.mlc import first_zone_contact, solve_mlc
from .cedi.vehicle import solve as solve_vehicle
from .search import candidate_grids


def _scene(ego, opponent, surrounding):
    actors = tuple(surrounding)
    all_states = (ego,) + (() if opponent is None else (opponent,)) + actors
    if not all(isinstance(s, VehicleState) for s in all_states):
        raise TypeError('all actors must be VehicleState instances')
    ids = [s.vid for s in all_states]
    if len(set(ids)) != len(ids):
        raise ValueError('actor IDs must be unique; surrounding must exclude the evaluated pair')
    return actors


def _rectangle(state):
    return Vehicle(str(state.vid), (state.x, state.y), (state.vx, state.vy),
                   state.length, state.width, math.degrees(state.heading))


def detect_vehicle_conflict(a: VehicleState, b: VehicleState,
                            surrounding: Iterable[VehicleState] = ()) -> RCDResult:
    """P1 plus scene P2, using every supplied surrounding rectangle.

    No forward-distance or ET threshold is silently applied. Initial A-C/B-C
    overlap is returned as unresolved instead of being called reachable.
    """
    others = _scene(a, b, surrounding)
    av, bv = _rectangle(a), _rectangle(b)
    et = swept_contact(av, bv)
    args = dict(subject_vehicle=a.vid, opponent_vehicle=b.vid, initial_et=et)
    if et is None:
        return RCDResult('no_predicted_contact', predicted_contact=False, reachable=False, **args)
    if et <= EPS:
        return RCDResult('initial_contact', predicted_contact=True, reachable=True, **args)
    unresolved = tuple(c.vid for c in others
                       if swept_contact(av, _rectangle(c)) == 0.
                       or swept_contact(bv, _rectangle(c)) == 0.)
    if unresolved:
        return RCDResult('unresolved_initial_surrounding_contact', predicted_contact=True,
                         reachable=None, unresolved_vehicles=unresolved, **args)
    try:
        prepared = rcd.prepare_ab(av, bv)
        pairs = list(prepared.pairs)
        blockers = []
        per_vehicle = {}
        for c in others:
            cv = _rectangle(c)
            detail = rcd.evaluate_prepared_c(prepared, cv)
            per_vehicle[c.vid] = detail
            if detail['blocks']:
                blockers.append(c.vid)
    except (ValueError, RuntimeError, ZeroDivisionError) as exc:
        return RCDResult('geometry_error', predicted_contact=True, reachable=None,
                         diagnostics={'error': str(exc)}, **args)
    return RCDResult('blocked' if blockers else 'reachable', predicted_contact=True,
                     reachable=not blockers, blockers=tuple(blockers),
                     diagnostics={'P1': True, 'P2': not blockers, 'Q': not blockers,
                                  'pairs': pairs, 'A_c': list(prepared.spatial.a_c),
                                  'B_c': list(prepared.spatial.b_c),
                                  'geometry_method': prepared.spatial.method,
                                  'SC_rule': rcd.SC_RULE,
                                  'surrounding': per_vehicle}, **args)


@lru_cache(maxsize=8)
def _grids(config):
    grids = candidate_grids(config, config.expansion_factors)
    for grid in grids:
        grid.flags.writeable = False
    return grids


def evaluate_mlc(ego: VehicleState, surrounding: Iterable[VehicleState] = (),
                 config: MLCConfig | None = None) -> CEDIResult:
    """Quantify MLC from a single vehicle perspective under the original model."""
    cfg = MLCConfig() if config is None else config
    if not isinstance(cfg, MLCConfig):
        raise TypeError('config must be MLCConfig')
    others = _scene(ego, None, surrounding)
    raw = solve_mlc(ego, list(others), cfg)
    initial_et = first_zone_contact(ego, 0., 0., cfg)
    initial_et = initial_et if math.isfinite(initial_et) else None
    # Missing / failed states remain unscored. Only no predicted contact is zero.
    cedi = raw.get('CEDI', 0. if raw['status'] == 'no_predicted_work_zone_contact' else None)
    scored = raw['status'] == 'feasible'
    return CEDIResult(raw['status'], cedi, ego.vid, None, initial_et,
                      raw.get('ax_mps2'), raw.get('ay_mps2'),
                      raw.get('a_front_mps2') if scored else None,
                      raw.get('a_s_mps2') if scored else None,
                      diagnostics=raw)


def evaluate_vehicle_conflict(ego: VehicleState, opponent: VehicleState,
                              surrounding: Iterable[VehicleState] = (),
                              config: VehicleConfig | None = None,
                              context: StaticEnvironment | None = None) -> CEDIResult:
    """Score ego; reverse ego/opponent to score the other perspective.

    This function assumes a pair has been selected for scoring. Call
    detect_vehicle_conflict separately if P2 scene screening is required.
    A/B keep their observed headings; surrounding responses are not mirrored.
    """
    cfg = VehicleConfig() if config is None else config
    env = StaticEnvironment() if context is None else context
    if not isinstance(cfg, VehicleConfig) or not isinstance(env, StaticEnvironment):
        raise TypeError('config must be VehicleConfig and context must be StaticEnvironment')
    others = _scene(ego, opponent, surrounding)
    raw = solve_vehicle(ego, opponent, list(others), env.as_array(), cfg,
                        _grids(cfg), cfg.expansion_factors, sample_s=cfg.sample_s,
                        include_tail=cfg.include_tail,
                        expand_response_limits=cfg.expand_response_limits, audit=cfg.audit)
    responses = tuple(raw.get('responses', ()))
    scored = raw['status'] == 'feasible'
    front = sum(r['total_demand_mps2'] for r in responses if r['role'] == 'front') if scored else None
    rear = sum(r['total_demand_mps2'] for r in responses if r['role'] == 'rear') if scored else None
    return CEDIResult(raw['status'], raw['CEDI'], ego.vid, opponent.vid,
                      raw['original_ET_s'], raw['ax_mps2'], raw['ay_mps2'],
                      front, rear, responses, raw)


def evaluate_vehicle_pair(a: VehicleState, b: VehicleState,
                          surrounding: Iterable[VehicleState] = (),
                          config: VehicleConfig | None = None,
                          context: StaticEnvironment | None = None) -> dict[int, CEDIResult]:
    """Return both vehicle perspectives without assuming symmetric costs."""
    others = tuple(surrounding)
    return {a.vid: evaluate_vehicle_conflict(a, b, others, config, context),
            b.vid: evaluate_vehicle_conflict(b, a, others, config, context)}
