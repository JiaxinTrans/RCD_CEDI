"""Two test Examples: A-B spatial interaction (P1) and reachability with C (P2)."""
import math
from rcd_cedi.geometry import Vehicle, spatial_interaction
from rcd_cedi.rcd import prepare_ab, evaluate_prepared_c

# Arguments: ID, x, y, speed (m/s), heading (rad), length (m), width (m).
a = Vehicle.from_paper_state("A", 0, 0, 10, 0, 4, 2)
b = Vehicle.from_paper_state("B", 20, 0, 5, 0, 4, 2)


# ============================================================
# Example 1: Check P1 for A and B without surrounding vehicles.
# ============================================================
print("\nExample 1: A-B spatial interaction (P1)")

p1_cases = [
    ("A approaches B", b),
    (
        "A and B are laterally separated",
        Vehicle.from_paper_state("B", 20, 5, 5, 0, 4, 2),
    ),
    (
        "B moves faster than A",
        Vehicle.from_paper_state("B", 20, 0, 12, 0, 4, 2),
    ),
]

for label, other in p1_cases:
    geometry = spatial_interaction(a, other)

    print(f"\nScenario: {label}")
    print(f"P1: {geometry.p1}")
    print(f"ET: {geometry.contact_time}")


# ============================================================
# Example 2: Place C between A and B and check P2.
# A travels at 10 m/s and B at 5 m/s, giving ET_AB = 3.2 s.
# C faces and moves along +y; its clearing distance is 3 m
# ============================================================
print("\nExample 2: A-B reachability with an intermediate vehicle C")

prepared = prepare_ab(a, b)
print(f"A-B contact time: {prepared.t_ab} s")

# Test stationary C, clearance after ET, at ET, and before ET.
for speed in (0.0, 0.5, 0.9375, 1.5):
    c = Vehicle.from_paper_state(
        "C",
        10, 0,
        speed,
        math.pi / 2,
        4, 2,
    )

    result = evaluate_prepared_c(prepared, c)

    print(f"\nC speed: {speed} m/s")
    print(f"I_C: {result['I_C']}")
    print(f"S_C: {result['S_C']}")
    print(f"D_C: {result['D_C']} m")
    print(f"ALT_C: {result['ALT_C']} s")
    print(f"P1: {result['P1']}")
    print(f"P2: {result['P2']}")
    print(f"Q = P1 AND P2: {result['Q']}")

    if result["blocks"]:
        print("Result: C blocks the A-B conflict; it is unreachable.")
    else:
        print("Result: C does not block the A-B conflict; it is reachable.")


