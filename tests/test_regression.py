"""Compare migrated solvers against outputs from the supplied original code."""
import json
import math
from pathlib import Path

import pytest

from rcd_cedi import (Environment, MLCConfig, VehicleConfig, VehicleState,
                      evaluate_mlc, evaluate_vehicle_conflict)

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/source_regression.json').read_text())


def assert_same(actual, expected):
    if isinstance(expected, dict):
        actual = {str(k): v for k, v in actual.items() if k != 'elapsed_ms'}
        expected = {str(k): v for k, v in expected.items() if k != 'elapsed_ms'}
        assert actual.keys() == expected.keys()
        for k in expected:
            assert_same(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_same(a, e)
    elif isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual)
        elif math.isinf(expected):
            assert actual == expected
        else:
            assert actual == pytest.approx(expected, abs=1e-10, rel=1e-10)
    else:
        assert actual == expected


@pytest.mark.parametrize('case', FIXTURE['vehicle_cases'], ids=lambda c: c['name'])
def test_vehicle_source_regression(case):
    config = VehicleConfig(**case['config'], expansion_factors=tuple(case['factors']))
    result = evaluate_vehicle_conflict(
        VehicleState(**case['a']), VehicleState(**case['b']),
        [VehicleState(**s) for s in case['surrounding']], config,
        Environment(tuple(map(tuple, case['polygon']))))
    assert_same(result.diagnostics, case['expected'])
    if result.status == 'feasible':
        assert result.diagnostics['planning_duration_s'] == result.initial_et
        assert result.diagnostics['retained_action_audit'] == 'passed'


@pytest.mark.parametrize('case', FIXTURE['mlc_cases'], ids=lambda c: c['name'])
def test_mlc_source_regression(case):
    result = evaluate_mlc(VehicleState(**case['ego']),
                          [VehicleState(**s) for s in case['surrounding']],
                          MLCConfig(**case['config']))
    assert_same(result.diagnostics, case['expected'])


def test_vehicle_without_obstacle_matches_far_obstacle():
    case = FIXTURE['vehicle_cases'][0]
    config = VehicleConfig(**case['config'], expansion_factors=tuple(case['factors']))
    result = evaluate_vehicle_conflict(VehicleState(**case['a']),
                                       VehicleState(**case['b']), config=config)
    assert_same(result.diagnostics, case['expected'])
