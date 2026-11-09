"""Independent equation / edge-case checks against manuscript Eqs. (1)-(16)."""
import math

import numpy as np
import pytest

from rcd_cedi.geometry import (Vehicle, Candidate, add, sub, mul, dot, cross,
    closest_collision_sets, vertex_edge_candidates, ray_edge_candidates,
    spatial_interaction, contact_time_from_pairs, swept_contact)
from rcd_cedi import geometry, rcd


def pair():
    return (Vehicle.from_paper_state('A',0,0,10,0,4,2),
            Vehicle.from_paper_state('B',20,0,0,0,4,2))


def test_eq1_canonical_state_and_angle_units():
    a=Vehicle.from_paper_state('A',3,4,10,math.pi/6,4,2)
    np.testing.assert_allclose(a.velocity,(5*math.sqrt(3),5))
    assert a.heading_deg == pytest.approx(30)
    assert a.paper_aligned
    assert not Vehicle('general',(0,0),(0,10),4,2,0).paper_aligned


@pytest.mark.parametrize('angle,expected',[
    (0.,((2,1),(2,-1),(-2,-1),(-2,1))),
    (math.pi/2,((-1,2),(1,2),(1,-2),(-1,-2))),
])
def test_eq3_numbered_corners(angle,expected):
    a=Vehicle.from_paper_state('A',0,0,0,angle,4,2)
    np.testing.assert_allclose(a.corners(),expected,atol=1e-12)


def test_eq4_to_6_ray_parameter_and_relative_frame():
    a=Vehicle.from_paper_state('A',0,0,10,.12,4,2)
    b=Vehicle.from_paper_state('B',20,2,2,-.1,5,2)
    actors={a.name:a,b.name:b}
    candidates=vertex_edge_candidates(a,b)
    assert candidates
    for c in candidates:
        moving,fixed=actors[c.vertex_vehicle],actors[c.edge_vehicle]
        relative=sub(moving.velocity,fixed.velocity)
        start=fixed.corners()[c.edge_index-1]
        end=fixed.corners()[c.edge_index%4]
        assert c.time>=0 and -1e-9<=c.edge_fraction<=1+1e-9
        np.testing.assert_allclose(add(c.V,mul(c.time,relative)),c.E,atol=1e-10)
        np.testing.assert_allclose(add(start,mul(c.edge_fraction,sub(end,start))),c.E,atol=1e-10)
        # Both participants reach the same WORLD point at t.
        np.testing.assert_allclose(add(c.V,mul(c.time,moving.velocity)),
                                   add(c.E,mul(c.time,fixed.velocity)),atol=1e-10)


def test_eq7_9_15_closest_sets_and_retained_ties():
    a,b=pair()
    ac,bc=closest_collision_sets(a,b)
    assert ac and bc
    result=spatial_interaction(a,b)
    assert result.p1 and len(result.pairs)>=2
    assert set(result.pairs)<=set(ac+bc)
    for selected in result.pairs:
        assert selected.distance==pytest.approx(16.)
    assert contact_time_from_pairs(result.pairs,10.)==pytest.approx(1.6)
    assert result.contact_time==pytest.approx(1.6)


def test_sat_degeneracy_completion_is_explicit(monkeypatch):
    a,b=pair()
    # Exercise the documented missing-ray completion independently of P2.
    monkeypatch.setattr(geometry,'ray_edge_candidates',lambda *args: [])
    result=geometry.spatial_interaction(a,b)
    assert result.p1 and result.pairs
    assert result.method=='sat_degenerate_contact_completion'
    assert not result.a_c and not result.b_c
    assert contact_time_from_pairs(result.pairs,10.)==pytest.approx(1.6)


def test_eq16_paper_aligned_equality_and_nearby_times():
    a,b=pair()
    for target_alt,expected_block in [(1.6-1e-6,False),(1.6,True),(1.6+1e-6,True)]:
        # C points along +y, giving lateral half-extent 2 and D_C=3.
        c=Vehicle.from_paper_state('C',10,0,3/target_alt,math.pi/2,4,2)
        result=rcd.rcd(a,b,c)
        assert result['ALT_C']==pytest.approx(target_alt)
        assert result['blocks'] is expected_block
        assert all(result['paper_state_aligned'].values())
        p=rcd.prepare_ab(a,b)
        assert rcd.c_blocks_prepared_ab(b,c,p.theta_ab,p.pairs,p.t_ab,c.corners()) is expected_block


def test_eq16_numerical_equality_guard_is_not_a_physical_threshold():
    assert not rcd.clears_before_contact(1.6-1e-11,1.6)
    assert rcd.clears_before_contact(1.6-1e-7,1.6)
    assert not rcd.clears_before_contact(math.inf,1.6)


def test_sc_global_region_convention_is_explicit():
    # A narrow C lies BETWEEN two paths, without touching either individual
    # line. The retained source convention treats the path region as occupied.
    pairs=[Candidate('A',1,'B',1,(2,-1),(18,-1),1.6),
           Candidate('A',2,'B',2,(2,1),(18,1),1.6)]
    b=Vehicle('B',(20,0),(0,0),4,2)
    c=Vehicle('C',(10,0),(0,0),4,.5)
    result=rcd.evaluate_c_obstruction(b,c,(1,0),pairs,1.6)
    assert result['I_C']==result['S_C']==1
    assert result['SC_rule']=='global_path_region_signs'
    assert result['blocks']


@pytest.mark.parametrize('velocity,label',[
    ((0.,0.),'relative_stationary'),
    ((1.,0.),'parallel_same_direction'),
    ((-1.,0.),'parallel_opposite_direction'),
])
def test_zero_transverse_speed_completion(velocity,label):
    a,b=pair()
    c=Vehicle('C',(10,0),velocity,4,2)
    result=rcd.rcd(a,b,c)
    assert result['zero_clearing_speed_case']==label
    assert result['ALT_C']==math.inf and result['blocks']


def test_random_paper_ray_equations_crosschecked_by_independent_sat():
    rng=np.random.default_rng(723)
    compared=0
    for k in range(400):
        aa,bb=rng.uniform(-math.pi,math.pi,2)
        va,vb=rng.uniform(1,20,2)
        a=Vehicle.from_paper_state('A',0,0,va,aa,4,2)
        relative=sub(a.velocity,(vb*math.cos(bb),vb*math.sin(bb)))
        time=rng.uniform(.5,5)
        offset=rng.normal(0,2,2)
        b=Vehicle.from_paper_state('B',relative[0]*time+offset[0],
                                 relative[1]*time+offset[1],vb,bb,5,2)
        sat=swept_contact(a,b)
        if sat is not None and sat<=1e-9:
            continue
        result=spatial_interaction(a,b)
        assert result.p1==(sat is not None)
        if sat is not None:
            compared+=1
            assert contact_time_from_pairs(result.pairs,math.hypot(*relative))==pytest.approx(sat,abs=1e-7)
    assert compared>100
