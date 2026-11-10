"""Shared CSV input, configuration, and export helpers; no CLI entry point."""
import csv
import json
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rcd_cedi import MLCConfig, StaticEnvironment, VehicleConfig, VehicleState, __version__

INPUT_FIELDS = ("time_s", "vid", "x", "y", "speed", "heading", "length", "width")
OPTIONAL_INPUT_FIELDS = ("body_heading",)
OUTPUT_FIELDS = (
    "time_s", "mode", "subject_vehicle", "opponent_vehicle", "status",
    "rcd_status", "p1", "p2_checked", "predicted_contact", "reachable",
    "initial_et", "blockers", "unresolved_vehicles", "surrounding_count",
    "surrounding_ids", "cedi", "ax", "ay", "front_demand", "rear_demand",
    "waiting_time_s",
)


def load_settings(path, mode):
    settings = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("configuration must be a JSON object")
    allowed = {"columns", mode} | ({"static_polygon"} if mode == "vehicle" else set())
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"unknown configuration sections for {mode}: {sorted(unknown)}")
    mapping = settings.get("columns", {})
    all_fields = INPUT_FIELDS + OPTIONAL_INPUT_FIELDS
    if not isinstance(mapping, dict) or set(mapping) - set(all_fields):
        raise ValueError("columns must map speed/heading input fields to CSV headers; vx/vy are derived")
    columns = {key: mapping.get(key, key) for key in all_fields}
    if any(not isinstance(value, str) or not value for value in columns.values()):
        raise ValueError("CSV column names must be nonempty strings")
    if len(set(columns.values())) != len(columns):
        raise ValueError("each input field must map to a different CSV column")
    cls = MLCConfig if mode == "mlc" else VehicleConfig
    config = cls(**settings.get(mode, {}))
    context = StaticEnvironment(settings.get("static_polygon"))
    return columns, config, context


def load_frames(path, columns):
    """Group exact decimal timestamps; validate every state before evaluation."""
    frames = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("CSV has duplicate column headers")
        if {"vx", "vy"}.intersection(headers):
            raise ValueError("vx/vy are derived, not CSV inputs; supply speed and heading instead")
        missing = {columns[key] for key in INPUT_FIELDS} - set(headers)
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for row in reader:
            try:
                time = Decimal(row[columns["time_s"]])
                vid = Decimal(row[columns["vid"]])
                if not time.is_finite():
                    raise ValueError("time_s must be finite")
                if not vid.is_finite() or vid != vid.to_integral_value():
                    raise ValueError("vid must be an integer")
                fields = {key: float(row[columns[key]]) for key in INPUT_FIELDS[2:]}
                for key in OPTIONAL_INPUT_FIELDS:
                    value = row.get(columns[key])
                    if value is not None and value.strip():
                        fields[key] = float(value)
                state = VehicleState(
                    vid=int(vid),
                    **fields,
                )
                frame = frames.setdefault(time, {})
                if state.vid in frame:
                    raise ValueError(f"duplicate vehicle ID {state.vid} at time {time}")
                frame[state.vid] = state
            except (ValueError, TypeError, KeyError, InvalidOperation) as exc:
                raise ValueError(f"CSV line {reader.line_num}: {exc}") from exc
    if not frames:
        raise ValueError("CSV contains no vehicle states")
    return frames


def make_row(time, mode, subject, opponent, surrounding, result=None, detection=None):
    row = dict.fromkeys(OUTPUT_FIELDS)
    row.update(time_s=str(time), mode=mode, subject_vehicle=subject,
               opponent_vehicle=opponent, surrounding_count=len(surrounding),
               surrounding_ids=";".join(str(v.vid) for v in surrounding))
    if detection is not None:
        row.update(rcd_status=detection.status, p1=detection.predicted_contact,
                   p2_checked=detection.status in ("reachable", "blocked"),
                   predicted_contact=detection.predicted_contact,
                   reachable=detection.reachable, initial_et=detection.initial_et,
                   blockers=";".join(map(str, detection.blockers)),
                   unresolved_vehicles=";".join(map(str, detection.unresolved_vehicles)))
    if result is not None:
        row.update({key: value for key, value in result.to_dict().items()
                    if key in row and key not in ("blockers", "unresolved_vehicles")})
        row["waiting_time_s"] = result.diagnostics.get("tau_s")
    return row


def check_output_paths(input_path, output_path, config_path):
    output = Path(output_path)
    metadata = Path(str(output) + ".config.json")
    protected = {Path(input_path).resolve()}
    if config_path is not None:
        protected.add(Path(config_path).resolve())
    if output.resolve() in protected or metadata.resolve() in protected:
        raise ValueError("output must not overwrite the input CSV or configuration")


def write_outputs(rows, output_path, mode, config, context, columns, skipped, **selection):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(mode=mode, code_version=__version__, columns=columns,
                    state_input="speed_heading", angle_unit="radians",
                    derived_velocity="vx = speed*cos(heading); vy = speed*sin(heading)",
                    body_heading_default="heading; optional body_heading overrides rectangle orientation only",
                    configuration=asdict(config), static_polygon=context.polygon,
                    p2_enabled=mode == "vehicle", timestamp_grouping="exact decimal values",
                    frames_without_subject=skipped, output_rows=len(rows), **selection)
    Path(str(path) + ".config.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return rows, metadata
