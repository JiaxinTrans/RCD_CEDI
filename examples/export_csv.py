"""Export RCD or CEDI results to CSV.

Run from the project root after installation:
    python examples/export_csv.py --output result.csv

The runnable demo uses RCD. Reuse save_results() with a CEDI result,
an iterable of result objects, or the dictionary from evaluate_vehicle_pair().
"""

import argparse
import csv
from pathlib import Path

from rcd_cedi import CEDIResult, RCDResult, VehicleState, detect_vehicle_conflict


FIELDS = [
    "status", "subject_vehicle", "opponent_vehicle",
    "initial_et", "predicted_contact", "reachable",
    "cedi", "ax", "ay", "front_demand", "rear_demand",
]


def save_results(results, output="result.csv"):
    """Write main scalar fields; None and inapplicable fields become blanks.

    Detailed responses, diagnostics, and vehicle-ID lists are not exported.
    """
    if isinstance(results, (RCDResult, CEDIResult)):
        results = [results]
    elif isinstance(results, dict):
        results = results.values()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="result.csv", help="Output CSV path")
    args = parser.parse_args()

    a = VehicleState(1, 0, 0, 10, 0, 0, 4, 2)
    b = VehicleState(2, 20, 0, 0, 0, 0, 4, 2)
    c = VehicleState(3, 10, 0, 0, 1, 0, 4, 2)
    result = detect_vehicle_conflict(a, b, surrounding=[c])
    path = save_results(result, args.output)
    print(f"Saved {result.status} result to {path}")


if __name__ == "__main__":
    main()
