"""Run after: python -m pip install -e ."""
import math
from rcd_cedi import VehicleState, detect_vehicle_conflict

# Arguments: ID, x, y, speed, heading (rad), length, width.
a = VehicleState(1, 0, 0, 10, 0, 4, 2)
b = VehicleState(2, 20, 0, 0, 0, 4, 2)
for speed in (0.0, 1.0, 1.25, 2.0):
    # Preserve the original example: C translates along +y but its body faces +x.
    c = VehicleState(3, 10, 0, speed, math.pi/2, 4, 2, body_heading=0)
    result = detect_vehicle_conflict(a, b, [c])
    print(f'C speed={speed} m/s: {result.status}, ET={result.initial_et}')
