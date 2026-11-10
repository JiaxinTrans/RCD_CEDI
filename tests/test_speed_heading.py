"""Speed/heading contract, CSV input and fixed-body prediction invariants."""
import csv
import inspect
import math
from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal

import numpy as np
import pytest

from csv_tools import _common, mlc, vehicle
from rcd_cedi import VehicleState, VehicleConfig, detect_vehicle_conflict
from rcd_cedi.api import _rectangle
from rcd_cedi.models import _at_motion
from rcd_cedi.static import pack_scene


@pytest.mark.parametrize('speed,heading', [
    (10, 0), (10, math.pi/2), (10, math.pi), (10, -math.pi/2),
    (7, .37), (0, 1.2), (4, 7*math.pi),
])
def test_velocity_is_derived_and_not_an_input(speed, heading):
    state = VehicleState(1, 2, 3, speed, heading, 4, 2)
    assert state.vx == speed*math.cos(heading)
    assert state.vy == speed*math.sin(heading)
    assert math.hypot(state.vx, state.vy) == pytest.approx(speed)
    assert state.body_angle == heading
    assert not {'vx', 'vy'} & set(inspect.signature(VehicleState).parameters)
    assert not {'vx', 'vy'} & set(asdict(state))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        state.vx = 99
    with pytest.raises((FrozenInstanceError, AttributeError)):
        state.vy = 99
    with pytest.raises(TypeError):
        VehicleState(vid=1, x=2, y=3, vx=10, vy=0, heading=0, length=4, width=2)


@pytest.mark.parametrize('invalid', [
    {'speed': -1}, {'speed': math.inf}, {'speed': math.nan},
    {'heading': math.nan}, {'body_heading': math.inf},
])
def test_invalid_polar_state_rejected(invalid):
    values = dict(vid=1, x=0, y=0, speed=10, heading=0, length=4, width=2)
    values.update(invalid)
    with pytest.raises(ValueError):
        VehicleState(**values)


def test_replace_recomputes_velocity_and_default_body_heading():
    initial = VehicleState(1, 0, 0, 10, 0, 4, 2)
    assert initial.vx == 10  # Populate the cached derived values.
    changed = replace(initial, speed=5, heading=math.pi/2)
    assert changed.vy == pytest.approx(5)
    assert changed.vx == pytest.approx(0, abs=1e-12)
    assert changed.body_angle == math.pi/2
    assert initial.vx == 10 and initial.body_angle == 0


def test_separate_body_heading_reaches_geometry_and_packing():
    state = VehicleState(1, 0, 1.75, 10, math.pi/2, 4, 2, body_heading=0)
    assert state.direction == pytest.approx((0, 1), abs=1e-12)
    assert state.basis[0] == (1, 0)
    assert state.theta == state.body_angle == 0
    rectangle = _rectangle(state)
    assert rectangle.heading_deg == 0
    assert rectangle.velocity == pytest.approx((0, 10), abs=1e-12)
    other = VehicleState(2, 20, 1.75, 0, 0, 4, 2)
    _, packed, *_ = pack_scene(state, other, [], None, VehicleConfig())
    np.testing.assert_allclose(packed[0, 3:5], (state.vx, state.vy))
    np.testing.assert_allclose(packed[0, 5:7], (1, 0))


def test_predicted_acceleration_does_not_rotate_fixed_body():
    initial = VehicleState(1, 0, 1.75, 10, .2, 4, 2)
    predicted = _at_motion(initial, (12, 2, 3, 4))
    assert predicted.speed == 5
    assert predicted.heading == math.atan2(4, 3)
    assert predicted.body_angle == initial.body_angle
    assert predicted.theta == initial.theta
    np.testing.assert_allclose((predicted.vx, predicted.vy), (3, 4), atol=1e-14)
    np.testing.assert_allclose(predicted.basis, initial.basis)


def test_rotating_aligned_rectangles_uses_heading_for_motion_and_shape():
    a = VehicleState(1, 0, 0, 10, 0, 4, 2)
    b = VehicleState(2, 20, 0, 0, 0, 4, 2)
    angle = .48
    rotated_a = replace(a, heading=angle)
    rotated_b = replace(b, x=20*math.cos(angle), y=20*math.sin(angle), heading=angle)
    initial = detect_vehicle_conflict(a, b)
    result = detect_vehicle_conflict(rotated_a, rotated_b)
    assert result.status == initial.status == 'reachable'
    assert result.initial_et == pytest.approx(initial.initial_et)


def test_csv_optional_body_heading_and_column_mapping(tmp_path):
    source = tmp_path/'scene.csv'
    source.write_text('time_s,vid,x,y,speed,heading,length,width,BODY\n'
                      '0.10,1,0,1.75,5,0.4,4,2,0\n'
                      '0.100,2,20,1.75,0,0.3,4,2,\n')
    import json
    settings = tmp_path/'config.json'
    settings.write_text(json.dumps({'columns': {'body_heading': 'BODY'}}))
    columns, *_ = _common.load_settings(settings, 'vehicle')
    frame = _common.load_frames(source, columns)[Decimal('.1')]
    assert frame[1].body_angle == 0 and frame[1].heading == .4
    assert frame[1].vx == 5*math.cos(.4)
    assert frame[2].body_heading is None and frame[2].body_angle == .3


def test_csv_requires_speed_and_heading_and_rejects_component_input(tmp_path):
    columns, *_ = _common.load_settings(None, 'mlc')
    source = tmp_path/'old.csv'
    source.write_text('time_s,vid,x,y,vx,vy,heading,length,width\n0,1,0,1,10,0,0,4,2\n')
    with pytest.raises(ValueError, match='derived'):
        _common.load_frames(source, columns)
    source.write_text('time_s,vid,x,y,heading,length,width\n0,1,0,1,0,4,2\n')
    with pytest.raises(ValueError, match='speed'):
        _common.load_frames(source, columns)
    source.write_text('time_s,vid,x,y,speed,heading,length,width\n0,1,0,1,-1,0,4,2\n')
    with pytest.raises(ValueError, match='nonnegative'):
        _common.load_frames(source, columns)


@pytest.mark.parametrize('workflow', ['mlc', 'vehicle'])
def test_csv_exports_derived_input_contract(workflow, tmp_path):
    source = tmp_path/'scene.csv'
    source.write_text('time_s,vid,x,y,speed,heading,length,width\n'
                      '0,1,0,8.75,10,0,4,2\n0,2,20,5.25,0,0,4,2\n')
    if workflow == 'mlc':
        _, meta = mlc.run(source, tmp_path/'out.csv', ego=1)
    else:
        _, meta = vehicle.run(source, tmp_path/'out.csv', ego=1, other=2)
    assert meta['state_input'] == 'speed_heading'
    assert meta['angle_unit'] == 'radians'
    assert not {'vx', 'vy'} & set(meta['columns'])
    assert meta['p2_enabled'] is (workflow == 'vehicle')
