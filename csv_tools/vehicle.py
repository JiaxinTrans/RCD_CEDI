"""Vehicle conflict CSV workflow: RCD P1/P2, CEDI, and integrated export.

Run from the repository root after installing rcd_cedi:
    python -m csv_tools.vehicle --input csv_tools/data/trajectories_vehicle.csv --ego 1 --other 2 --both

CSV columns: time_s,vid,x,y,speed,heading,length,width. Angles are radians.
Optional body_heading controls rectangle orientation only; velocity is derived.
"""
import argparse

from rcd_cedi import detect_vehicle_conflict, evaluate_vehicle_conflict, evaluate_vehicle_pair

if __package__:
    from ._common import check_output_paths, load_frames, load_settings, make_row, write_outputs
else:
    from _common import check_output_paths, load_frames, load_settings, make_row, write_outputs


def run(input_path, output_path="vehicle_results.csv", ego=1, other=2, config_path=None, both=False):
    if ego == other:
        raise ValueError("--ego and --other must differ")
    check_output_paths(input_path, output_path, config_path)
    columns, config, context = load_settings(config_path, "vehicle")
    frames = load_frames(input_path, columns)
    rows, skipped = [], 0
    for time, states in sorted(frames.items()):
        if ego not in states:
            skipped += 1
            continue
        surrounding = [v for vid, v in sorted(states.items()) if vid not in (ego, other)]
        if other not in states:
            row = make_row(time, "vehicle", ego, other, surrounding)
            row["status"] = "missing_opponent"
            rows.append(row)
            continue
        subject, opponent = states[ego], states[other]
        detection = detect_vehicle_conflict(subject, opponent, surrounding=surrounding)
        if detection.status != "reachable":
            pairs = [(ego, other), (other, ego)] if both else [(ego, other)]
            for a, b in pairs:
                row = make_row(time, "vehicle", a, b, surrounding, detection=detection)
                row["status"] = "skipped_rcd_" + detection.status
                rows.append(row)
            continue
        if both:
            results = evaluate_vehicle_pair(subject, opponent, surrounding, config=config, context=context)
        else:
            results = {ego: evaluate_vehicle_conflict(subject, opponent, surrounding,
                                                     config=config, context=context)}
        for result in results.values():
            rows.append(make_row(time, "vehicle", result.subject_vehicle, result.opponent_vehicle,
                                 surrounding, result=result, detection=detection))
    if not rows:
        raise ValueError(f"subject vehicle {ego} is absent from all frames")
    return write_outputs(rows, output_path, "vehicle", config, context, columns, skipped,
                         ego=ego, other=other, both=both)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Vehicle-conflict trajectory CSV")
    parser.add_argument("--ego", required=True, type=int)
    parser.add_argument("--other", required=True, type=int)
    parser.add_argument("--both", action="store_true", help="Score both vehicle perspectives")
    parser.add_argument("--output", default="vehicle_results.csv")
    parser.add_argument("--config", help="Optional vehicle JSON settings and column mapping")
    args = parser.parse_args()
    try:
        rows, metadata = run(args.input, args.output, args.ego, args.other, args.config, args.both)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {len(rows)} rows to {args.output}")
    print(f"Configuration: {args.output}.config.json")
    if metadata["frames_without_subject"]:
        print(f"Frames without subject: {metadata['frames_without_subject']}")


if __name__ == "__main__":
    main()
