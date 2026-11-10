"""Integration checks for the two CSV workflows."""
import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

from csv_tools import _common, mlc, vehicle
from rcd_cedi import evaluate_mlc, evaluate_vehicle_pair

ROOT = Path(mlc.__file__).parent


def write_states(path, states):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(_common.INPUT_FIELDS)
        writer.writerows(states)
    return path


def test_mlc_matches_direct_api_and_keeps_all_surrounding(tmp_path):
    source = ROOT / 'data/trajectories_mlc.csv'
    settings = ROOT / 'data/mlc_config.json'
    columns, cfg, _ = _common.load_settings(settings, 'mlc')
    frame = _common.load_frames(source, columns)[Decimal(0)]
    expected = evaluate_mlc(frame[1], [frame[2], frame[3]], config=cfg)
    rows, meta = mlc.run(source, tmp_path/'mlc.csv', 1, settings)
    assert len(rows) == 2 and rows[0]['surrounding_ids'] == '2;3'
    assert rows[0]['status'] == expected.status
    assert rows[0]['cedi'] == pytest.approx(expected.cedi)
    assert rows[0]['waiting_time_s'] == expected.diagnostics.get('tau_s')
    assert all(row['p1'] and not row['p2_checked'] and row['reachable'] is None for row in rows)
    assert meta['p2_enabled'] is False


def test_mlc_no_contact_and_initial_contact_are_distinct(tmp_path):
    source = write_states(tmp_path/'states.csv', [
        [0, 1, 0, 8.75, 10, 0, 4, 2],
        [1, 1, 50, 1.75, 10, 0, 4, 2],
    ])
    rows, _ = mlc.run(source, tmp_path/'out.csv', 1)
    assert rows[0]['p1'] is False and rows[0]['cedi'] == 0
    assert rows[0]['status'] == 'no_predicted_work_zone_contact'
    assert rows[1]['p1'] is True and rows[1]['cedi'] is None
    assert rows[1]['status'] == 'already_at_work_zone'
    with (tmp_path/'out.csv').open(encoding='utf-8-sig') as f:
        values=list(csv.DictReader(f))
    assert float(values[0]['cedi']) == 0 and values[1]['cedi'] == ''


def test_mlc_contact_beyond_horizon_is_not_filtered(tmp_path):
    source = write_states(tmp_path/'states.csv', [[0, 1, -100, 1.75, 1, 0, 4, 2]])
    rows, meta = mlc.run(source, tmp_path/'out.csv', 1)
    assert rows[0]['initial_et'] > meta['configuration']['horizon']
    assert rows[0]['p1'] is True and rows[0]['cedi'] is not None
    assert rows[0]['status'] == 'feasible'


def test_vehicle_rcd_blocks_then_clears_without_reusing_old_actors(tmp_path):
    source = write_states(tmp_path/'states.csv', [
        [0, 1, 0, 1.75, 10, 0, 4, 2],
        [0, 2, 20, 1.75, 0, 0, 4, 2],
        [0, 3, 10, 1.75, 0, 0, 4, 2],
        [1, 1, 0, 1.75, 10, 0, 4, 2],
        [1, 2, 20, 1.75, 0, 0, 4, 2],
    ])
    settings = tmp_path/'config.json'
    settings.write_text(json.dumps({'vehicle': {'ax_steps':3,'ay_steps':3,'sector_steps':3,'expansion_factors':[1.]}}))
    rows, _ = vehicle.run(source, tmp_path/'out.csv', 1, 2, settings, both=True)
    assert len(rows) == 4
    assert rows[0]['status'] == rows[1]['status'] == 'skipped_rcd_blocked'
    assert rows[0]['cedi'] is None and rows[0]['blockers'] == '3'
    assert rows[2]['rcd_status'] == 'reachable' and rows[2]['surrounding_count'] == 0
    assert all(row['p2_checked'] for row in rows)


def test_vehicle_both_perspectives_match_direct_api(tmp_path):
    source = ROOT/'data/trajectories_vehicle.csv'
    settings = ROOT/'data/vehicle_config.json'
    columns, cfg, context = _common.load_settings(settings, 'vehicle')
    frame = _common.load_frames(source, columns)[Decimal(0)]
    expected = evaluate_vehicle_pair(frame[1], frame[2], [frame[3]], config=cfg, context=context)
    rows, meta = vehicle.run(source, tmp_path/'out.csv', 1, 2, settings, True)
    assert len(rows) == 4 and meta['p2_enabled'] is True
    for row in rows[:2]:
        reference = expected[row['subject_vehicle']]
        assert row['status'] == reference.status
        assert row['cedi'] == pytest.approx(reference.cedi)
        assert row['ax'] == reference.ax and row['ay'] == reference.ay
    one, _ = vehicle.run(source, tmp_path/'one.csv', 1, 2, settings)
    assert len(one) == 2 and all(row['subject_vehicle']==1 for row in one)


def test_missing_vehicle_and_no_contact_do_not_become_zero_scores(tmp_path):
    source = write_states(tmp_path/'states.csv', [
        [0, 1, 0, 1.75, 10, 0, 4, 2],
        [1, 1, 0, 1.75, 10, 0, 4, 2],
        [1, 2, 20, 5.25, 0, 0, 4, 2],
    ])
    rows, _ = vehicle.run(source, tmp_path/'out.csv', 1, 2)
    assert rows[0]['status']=='missing_opponent' and rows[0]['cedi'] is None
    assert rows[1]['status']=='skipped_rcd_no_predicted_contact' and rows[1]['cedi'] is None
    with pytest.raises(ValueError, match='absent'):
        mlc.run(source, tmp_path/'absent.csv', 99)


def test_column_mapping_exact_grouping_and_coordinates(tmp_path):
    source=tmp_path/'states.csv'
    source.write_text('T,ID,X,Y,SPEED,H,L,W\n0.10,1,281.123,1.875,10,0.1,4,2\n0.100,2,300,1.875,5,0.1,4,2\n0.100001,1,281.124,1.875,10,0.1,4,2\n')
    settings=tmp_path/'settings.json'
    settings.write_text(json.dumps({'columns': dict(zip(_common.INPUT_FIELDS,('T','ID','X','Y','SPEED','H','L','W')))}))
    columns, *_=_common.load_settings(settings,'mlc')
    frames=_common.load_frames(source,columns)
    assert len(frames)==2 and len(frames[Decimal('0.1')])==2
    assert frames[Decimal('0.1')][1].x==281.123
    assert frames[Decimal('0.1')][1].heading==0.1


def test_duplicate_states_config_isolation_and_input_protection(tmp_path):
    source=write_states(tmp_path/'states.csv',[[0,1,0,1.75,10,0,4,2]]*2)
    columns, *_=_common.load_settings(None,'mlc')
    with pytest.raises(ValueError,match='duplicate vehicle'):
        _common.load_frames(source,columns)
    with pytest.raises(ValueError,match='overwrite'):
        mlc.run(source,source,1)
    with pytest.raises(ValueError,match='unknown configuration'):
        _common.load_settings(ROOT/'data/vehicle_config.json','mlc')
    with pytest.raises(ValueError,match='must differ'):
        vehicle.run(source,tmp_path/'out.csv',1,1)
