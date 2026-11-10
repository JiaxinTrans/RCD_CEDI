import math
import pytest

from rcd_cedi import VehicleState, detect_vehicle_conflict
from rcd_cedi.geometry import Vehicle, swept_contact, earliest_ab_pairs


def v(vid, x, y=0., speed=0., heading=0., body_heading=0.):
    return VehicleState(vid, x, y, speed, heading, 4., 2., body_heading=body_heading)


@pytest.mark.parametrize('speed,reachable,alt', [(0.,False,math.inf),(1.,False,2.),
                                            (1.25,False,1.6),(2.,True,1.)])
def test_p2_alt_boundary(speed, reachable, alt):
    result = detect_vehicle_conflict(v(1,0,speed=10),v(2,20),
                                     [v(3,10,speed=speed,heading=math.pi/2)])
    assert result.initial_et == pytest.approx(1.6)
    assert result.reachable is reachable
    detail = result.diagnostics['surrounding'][3]
    assert detail['I_C'] == detail['S_C'] == 1
    assert detail['ALT_C'] == pytest.approx(alt)


def test_initial_surrounding_overlap_is_unresolved():
    result = detect_vehicle_conflict(v(1,0,speed=10),v(2,20),[v(3,1)])
    assert result.status == 'unresolved_initial_surrounding_contact'
    assert result.reachable is None
    assert result.unresolved_vehicles == (3,)


def test_no_contact_and_initial_contact():
    assert not detect_vehicle_conflict(v(1,0,speed=10),v(2,20,y=4)).predicted_contact
    initial = detect_vehicle_conflict(v(1,0,speed=10),v(2,3))
    assert initial.status == 'initial_contact' and initial.initial_et == 0.


def test_coincident_edge_fallback_and_zero_relative_speed():
    a = Vehicle('a',(0.,0.),(10.,0.),4.,2.)
    b = Vehicle('b',(20.,0.),(0.,0.),4.,2.)
    t = swept_contact(a,b)
    assert earliest_ab_pairs(a,b,t)[1]
    c = Vehicle('c',(20.,0.),(10.,0.),4.,2.)
    assert swept_contact(a,c) is None


def test_common_rotation_and_translation_preserve_rcd():
    angle = .37
    c,s = math.cos(angle),math.sin(angle)
    def transform(state):
        return VehicleState(state.vid,c*state.x-s*state.y+100,
                            s*state.x+c*state.y-20,
                            state.speed,state.heading+angle,state.length,state.width,
                            body_heading=state.body_angle+angle)
    states = [v(1,0,speed=10),v(2,20),v(3,10,speed=1,heading=math.pi/2)]
    result = detect_vehicle_conflict(*map(transform,states[:2]),[transform(states[2])])
    assert result.status == 'blocked'
    assert result.initial_et == pytest.approx(1.6)
    assert result.diagnostics['surrounding'][3]['ALT_C'] == pytest.approx(2.)
