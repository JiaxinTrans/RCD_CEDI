"""Run after: python -m pip install -e ."""
from rcd_cedi import VehicleState, detect_vehicle_conflict

a = VehicleState(1, 0, 0, 10, 0, 0, 4, 2)
b = VehicleState(2, 20, 0, 0, 0, 0, 4, 2)
for vy in (0.0, 1.0, 1.25, 2.0):
    c = VehicleState(3, 10, 0, 0, vy, 0, 4, 2)
    result = detect_vehicle_conflict(a, b, [c])
    print(f'C vy={vy}: {result.status}, ET={result.initial_et}')
