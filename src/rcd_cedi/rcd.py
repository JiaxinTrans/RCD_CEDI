"""Paper-calibrated rectangle RCD, Section 3.2, Eqs. (2),(10)-(16).

I_C: Eq. (10), every C corner and every selected (V_i,E_i).
S_C: Eq. (11) global path-region sign convention retained from the supplied code.
     The uploaded PDF omits its full formula; Fig. 4(b) only illustrates one
     path. This convention is not labelled a numbered paper equation.
ALT: Eq. (12); d_C, signed offsets, D_C: Eqs. (13)-(15).
P2: Eq. (17), a C blocks if I_C*S_C==1 and ALT_C >= t_AB_c.
    Equality is BLOCKED. Strict clearance before first contact is required.

Zero transverse relative speed is completed as persistent occupancy after
I_C/S_C screening, including stationary and opposite-parallel cases for which
the PDF does not provide a finite ALT formula. See PAPER_ALIGNMENT.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import hypot, inf

from .constants import EPS, Vec
from .geometry import (Vehicle, Candidate, SpatialInteraction, add, sub, mul,
                       dot, cross, swept_contact, earliest_ab_pairs,
                       spatial_interaction, contact_time_from_pairs)

# Numerical equality guard in seconds / relative time, not a risk threshold.
TIME_ATOL = 1e-9
TIME_RTOL = 1e-9
SC_RULE = 'global_path_region_signs'


@dataclass(frozen=True)
class PreparedAB:
    """Cache A-B Eqs. (4)-(9),(15) once for all surrounding vehicles."""
    a: Vehicle
    b: Vehicle
    spatial: SpatialInteraction
    theta_ab: Vec | None

    @property
    def t_ab(self):
        return self.spatial.contact_time

    @property
    def pairs(self):
        return self.spatial.pairs


def prepare_ab(a: Vehicle, b: Vehicle) -> PreparedAB:
    """Prepare P1 / closest pairs; initial contact is outside positive-ET P2."""
    spatial = spatial_interaction(a, b)
    if spatial.initial_contact:
        raise ValueError('A and B overlap or touch initially')
    relative = sub(a.velocity, b.velocity)
    speed = hypot(*relative)
    theta = mul(1. / speed, relative) if spatial.p1 and speed > EPS else None
    return PreparedAB(a, b, spatial, theta)


def clears_before_contact(alt: float, t_ab: float) -> bool:
    """Eq. (16): ALT < ET; near equality is treated as equality."""
    return (math.isfinite(alt) and alt < t_ab
            and not math.isclose(alt, t_ab, rel_tol=TIME_RTOL, abs_tol=TIME_ATOL))


def evaluate_c_obstruction(b, c, theta, pairs, t_ab, c_corners=None) -> dict:
    """Shared P2 for a prepared pair; scene P2 requires no blocking C.

    Preconditions: positive ET, nonempty pairs, unit theta_ab, no initial
    A-C/B-C contact. The scene API checks initial contacts separately.
    """
    if not pairs or not math.isfinite(t_ab) or t_ab <= EPS:
        raise ValueError('P2 requires nonempty contact pairs and positive finite ET')
    if not math.isclose(hypot(*theta), 1., rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError('theta_ab must be a unit vector')
    corners = c.corners() if c_corners is None else c_corners
    if len(corners) != 4:
        raise ValueError('C must have four rectangle corners')

    # Eq. (10): all (i,m); EPS guards the strict zero boundary.
    products = tuple(dot(sub(cm, pair.V), theta) * dot(sub(cm, pair.E), theta)
                     for pair in pairs for cm in corners)

    i_c = all(value < -EPS for value in products)

    # Unnumbered S_C convention: global signs across all selected paths.
    # Fig. 4(b) uses dot(C_m-V_i,theta_perp). Cross reverses every sign under
    # a left normal, without changing a same-side/sign-change decision.
    sides = tuple(cross(sub(cm, pair.V), theta)
                  for pair in pairs for cm in corners)
    s_c = not (all(value > EPS for value in sides)
               or all(value < -EPS for value in sides))
    result = dict(
        ab_collision=True, t_ab_c=t_ab, pairs=list(pairs), theta_AB=theta,
        I_C=int(i_c), S_C=int(s_c), SC_rule=SC_RULE,
        ALT_C=None, D_C=None, d_C=None, d_VC_min=None, d_VC_max=None,
        w=None, blocks=False, reachable=True,
        projection_products=products, side_values=sides,
        zero_clearing_speed_case=None,
    )
    if not (i_c and s_c):
        result['reason'] = 'C fails I_C or S_C; it does not obstruct these paths'
        return result

    # Eq. (12): half-extent perpendicular to theta_AB.
    d_c = max(abs(cross(sub(cm, c.center), theta)) for cm in corners)
    # Eq. (13): signed offsets, NOT absolute distances.
    offsets = tuple(cross(sub(c.center, pair.V), theta) for pair in pairs)
    d_min, d_max = min(offsets), max(offsets)
    v_cb = sub(c.velocity, b.velocity)
    w = cross(v_cb, theta)
    # Eq. (14): distance to clear the last selected path with the entire body.
    if w > EPS:
        distance = max(0., d_c - d_min)
    elif w < -EPS:
        distance = max(0., d_c + d_max)
    else:
        distance = inf
        result['zero_clearing_speed_case'] = (
            'relative_stationary' if hypot(*v_cb) <= EPS else
            'parallel_same_direction' if dot(v_cb, theta) > 0. else
            'parallel_opposite_direction'
        )
    # Eq. (11), with explicit zero-speed completion.
    alt = distance / abs(w) if abs(w) > EPS else inf
    reachable = clears_before_contact(alt, t_ab)
    result.update(d_C=d_c, d_VC_min=d_min, d_VC_max=d_max,
                  w=w, D_C=distance, ALT_C=alt, blocks=not reachable,
                  reachable=reachable,
                  reason='Eq. (16): C blocks when ALT_C >= first A-B contact time')
    return result


def evaluate_prepared_c(prepared: PreparedAB, c: Vehicle) -> dict:
    """Detailed C check sharing prepared A-B geometry and Eq. (15)."""
    if not prepared.spatial.p1:
        return dict(ab_collision=False, reachable=False, blocks=False,
                    reason='Eq. (8): no predicted A-B contact')
    for actor in (prepared.a, prepared.b):
        if swept_contact(actor, c) == 0.:
            raise ValueError(f'{actor.name} and {c.name} overlap or touch initially')
    result = evaluate_c_obstruction(prepared.b, c, prepared.theta_ab,
                                    prepared.pairs, prepared.t_ab)
    result.update(
        P1=True, P2=result['reachable'], Q=result['reachable'],
        geometry_method=prepared.spatial.method,
        t_ab_from_eq15=contact_time_from_pairs(
            prepared.pairs, hypot(*sub(prepared.a.velocity, prepared.b.velocity))),
        paper_state_aligned={prepared.a.name: prepared.a.paper_aligned,
                             prepared.b.name: prepared.b.paper_aligned,
                             c.name: c.paper_aligned},
    )
    return result


def rcd(a: Vehicle, b: Vehicle, c: Vehicle) -> dict:
    """Compatibility API: P1 plus one C's P2, with formula diagnostics."""
    for first, second in ((a, b), (a, c), (b, c)):
        if swept_contact(first, second) == 0.:
            raise ValueError(f'{first.name} and {second.name} overlap or touch initially')
    return evaluate_prepared_c(prepare_ab(a, b), c)


def c_blocks_prepared_ab(b, c, theta, pairs, t_ab, c_corners) -> bool:
    """Boolean compatibility API sharing exactly the detailed P2 calculation."""
    return evaluate_c_obstruction(b, c, theta, pairs, t_ab, c_corners)['blocks']
