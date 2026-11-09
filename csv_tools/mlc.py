"""MLC CSV workflow: work-zone contact P1, CEDI evaluation, and CSV export.

Run from the repository root after installing rcd_cedi:
    python -m csv_tools.mlc --input csv_tools/data/trajectories_mlc.csv --ego 1

P1 reuses the existing finite work-zone boundary contact model. No P2 is
applied to MLC. Every same-frame surrounding vehicle remains in CEDI.
"""
import argparse
import math

from rcd_cedi import CEDIResult, evaluate_mlc
from rcd_cedi.cedi.mlc import first_zone_contact

if __package__:
    from ._common import check_output_paths, load_frames, load_settings, make_row, write_outputs
else:
    from _common import check_output_paths, load_frames, load_settings, make_row, write_outputs


def run(input_path, output_path="mlc_results.csv", ego=1, config_path=None):
    check_output_paths(input_path, output_path, config_path)
    columns, config, context = load_settings(config_path, "mlc")
    frames = load_frames(input_path, columns)
    rows, skipped = [], 0
    for time, states in sorted(frames.items()):
        if ego not in states:
            skipped += 1
            continue
        subject = states[ego]
        surrounding = [v for vid, v in sorted(states.items()) if vid != ego]
        # Constant-velocity vehicle/work-zone contact is the MLC P1 gate.
        # Do not impose a P2 test or a new ET/horizon acceptance threshold.
        et = first_zone_contact(subject, 0.0, 0.0, config)
        p1 = math.isfinite(et)
        if not p1:
            detection_status = "no_predicted_work_zone_contact"
            result = CEDIResult(detection_status, 0.0, ego, None, None)
        elif et == 0.0:
            detection_status = "initial_work_zone_contact"
            result = CEDIResult("already_at_work_zone", None, ego, None, 0.0)
        else:
            detection_status = "predicted_work_zone_contact"
            result = evaluate_mlc(subject, surrounding, config=config)
        row = make_row(time, "mlc", ego, None, surrounding, result=result)
        row.update(rcd_status=detection_status, p1=p1, predicted_contact=p1,
                   p2_checked=False)  # reachable remains blank: P2 is not applied.
        rows.append(row)
    if not rows:
        raise ValueError(f"subject vehicle {ego} is absent from all frames")
    return write_outputs(rows, output_path, "mlc", config, context, columns, skipped, ego=ego)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="MLC trajectory CSV")
    parser.add_argument("--ego", required=True, type=int, help="Subject vehicle ID")
    parser.add_argument("--output", default="mlc_results.csv")
    parser.add_argument("--config", help="Optional MLC JSON settings and column mapping")
    args = parser.parse_args()
    try:
        rows, metadata = run(args.input, args.output, args.ego, args.config)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved {len(rows)} rows to {args.output}")
    print(f"Configuration: {args.output}.config.json")
    if metadata["frames_without_subject"]:
        print(f"Frames without subject: {metadata['frames_without_subject']}")


if __name__ == "__main__":
    main()
