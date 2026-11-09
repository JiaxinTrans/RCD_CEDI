import math
from dataclasses import replace

import numpy as np
import pytest

from rcd_cedi import (Environment, MLCConfig, VehicleConfig, VehicleState,
                      detect_vehicle_conflict, evaluate_mlc,
                      evaluate_vehicle_conflict, evaluate_vehicle_pair)
from rcd_cedi.responses import sample_times
from rcd_cedi.search import candidate_grids


def state(vid=1,x=0.,y=1.875,vx=10.):
    return VehicleState(vid,x,y,vx,0.,0.,4.,2.)


@pytest.mark.parametrize('kwargs', [{'length':0},{'width':-1},{'x':math.nan},
                                     {'heading':math.inf},{'vid':True},{'vid':1.2}])
def test_invalid_state_rejected(kwargs):
    args=dict(vid=1,x=0,y=1,vx=10,vy=0,heading=0,length=4,width=2)
    args.update(kwargs)
    with pytest.raises(ValueError):
        VehicleState(**args)


def test_unique_ids_required_in_public_apis():
    a,b = state(),state(2,20,vx=0)
    with pytest.raises(ValueError):
        detect_vehicle_conflict(a,b,[a])
    with pytest.raises(ValueError):
        evaluate_mlc(a,[b,b])


def test_environment_validation():
    assert Environment().as_array() is None
    with pytest.raises(ValueError):
        Environment(((0,0),(2,2),(0,2),(2,0)))
    with pytest.raises(ValueError):
        Environment.rectangle(0,0,0,2)
    assert Environment.rectangle(0,2,0,2).as_array().shape == (4,2)


def test_nested_controls_keep_original_normalizers():
    cfg=VehicleConfig(ax_steps=9,ay_steps=9,sector_steps=3,expansion_factors=(1.,2.))
    small,large=candidate_grids(cfg,cfg.expansion_factors)
    assert set(map(tuple,small[:,1:3])) <= set(map(tuple,large[:,1:3]))
    for grid in (small,large):
        expected=np.hypot(grid[:,1]/np.where(grid[:,1]>0,cfg.ax_accel_g*9.81,cfg.ax_brake_g*9.81),grid[:,2]/(cfg.ay_g*9.81))
        np.testing.assert_allclose(grid[:,0],expected)


def test_samples_include_exact_ET():
    times=sample_times(1.63,.1)
    assert times[0]==0 and times[-1]==1.63 and np.all(np.diff(times)>0)


def test_pair_perspectives_and_generator_input():
    cfg=VehicleConfig(ax_steps=9,ay_steps=9,sector_steps=3,expansion_factors=(1.,2.))
    result=evaluate_vehicle_pair(state(),state(2,20,vx=0),(s for s in []),cfg)
    assert set(result)=={1,2}
    assert result[1].subject_vehicle==1 and result[2].subject_vehicle==2
    assert result[1].opponent_vehicle==2 and result[2].opponent_vehicle==1
    assert result[1].status=='feasible'


@pytest.mark.parametrize('kwargs',[{'gap':-.1},{'sample_s':0},{'expansion_factors':()},
                                   {'expansion_factors':(1.,.5)},{'ax_steps':1}])
def test_invalid_vehicle_config(kwargs):
    with pytest.raises(ValueError):
        VehicleConfig(**kwargs)


def test_shifted_mlc_boundary_explicitly_rejected():
    with pytest.raises(ValueError):
        MLCConfig(zone_start=(281,1))


def test_initial_static_overlap_unscored():
    cfg=VehicleConfig(ax_steps=9,ay_steps=9,sector_steps=3,expansion_factors=(1.,))
    result=evaluate_vehicle_conflict(state(),state(2,20,vx=0),config=cfg,
                                     context=Environment.rectangle(-3,3,0,4))
    assert result.status=='initial_SV_work_zone_overlap' and result.cedi is None
