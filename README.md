# RCD_CEDI

This repository implements a unified two-dimensional framework for risk quantification in highway work zones under intensive multi-vehicle interactions. The framework integrates Reachable Conflict Detection (RCD) and the Constrained Evasion Difficulty Index (CEDI), connecting conflict detectionn with vehicle risk assessment.

RCD evaluates both spatial interaction and conflict reachability, accounting for surrounding vehicles that may obstruct a predicted conflict. CEDI quantifies the minimum evasive difficulty required for conflict resolution within the available evasion time, considering static facility constraints and the maneuvering demands of the subject vehicle and surrounding multi-vehicles. Separate formulations address Mandatory Lane-Changing (MLC) and vehicle conflicts. For vehicle conflicts, CEDI evaluates each vehicle separately, capturing asymmetric risks arising from differences in surrounding constraints and available evasive options.&#x20;

## Research motivation

### Why RCD matters ?

Consider four vehicles traveling in the same lane, ordered from rear to front as A, B, C, and D, with decreasing speeds. Pairwise prediction may identify conflicts between A and all three vehicles ahead. However, B blocks A’s direct interactions with C and D. RCD checks whether each predicted conflict is reachable, excluding interactions blocked by surrounding vehicles.

### How do surrounding conditions shape risk?

The risk faced by a vehicle depends on constraints from both static facilities and surrounding vehicles. The following examples illustrate why CEDI considers these constraints.
<p align="center">
  <a href="asset/figs/limition.pdf">
    <img src="asset/figs/limition.png"
         alt="Figure 1. Overview of the framework"
         width="100%">
  </a>
</p>

<p align="left">
  <em>Figure Intensive vehicle interactions in work zones involve both static facility and multiple surrounding vehicles. Pair-based SSMs that assign a common risk value to an interacting vehicle pair can not fully capture the influence of these constraints on each vehicle’s risk.</em>
</p>

**Figure (a) Similar conflicts, different surroundings constraint**

Vehicles A and B approach the work zone in two separate scenarios. A has an open adjacent lane, while B’s lane-changing options are constrained by vehicle C. These additional constraints make conflict resolution more demanding, resulting in a higher risk for B.

**Figure (b) Different risks within the same interacting vehicle pair**

Vehicles A and B form an interacting vehicle pair but face different constraints.  A can slow down to avoid B. For B, braking may increase the risk of being struck by A, while the work zone and vehicle C restrict its lane-changing options.&#x20;

CEDI accounts for static and dynamic constraints and evaluates risk from each vehicle’s perspective, capturing both the influence of surrounding traffic and risk asymmetry within an interacting vehicle pair.

## Overview framework

[![Figure 1. Overview of the framework](asset/figs/framework.png)](asset/figs/framework.pdf)

The proposed 2D risk quantification framework for highway work zones comprises two stages. Stage 1 prepares vehicle state data and work zone geometry. Stage 2 quantifies risk through RCD to identify physically reachable conflicts and CEDI to evaluate risk under coupled static and dynamic mult-vehicles constraints in two conflict scenarios. Click the figure to view the full size.

## Case visualizations

Three cases from the main article illustrate conflict detection and risk evolution using historal trajectory data. The first visualization shows results from RCD and other conflict detection methods, while the second and third show risk evolution using CEDI and other safety measures. Each visualization includes the corresponding traffic scene. Two additional conflict cases are available in the [asset/gifs](asset/gifs).

[![Case 1: Conflict detection](asset/case_1.gif)

This case compares RCD with other conflict detection methods during an observed lane change by the subject vehicle in multi-vehicle traffic.

### Case 2: Aggressive overtaking and MLC
[![Case 2](asset/gifs/cedi_case1.gif)](asset/gifs/cedi_case1.gif?raw=true)

This case shows how CEDI captures emerging risk and reaches the top 20% threshold earlier, providing advance warning.

### Case 3: Conflicts with another vehicle and the work zone during a MLC

[![Case_3](asset/gifs/cedi_case2.gif)](asset/gifs/cedi_case2.gif?raw=true)

This case shows how CEDI evaluates the subject and interacting vehicles separately, accounting for their respective driving environments and capturing risk asymmetry.

## Installation

Python 3.10 or later is required. Open Windows PowerShell in the `RCD_CEDI` directory and run:

```powershell
# Create a virtual environment
python -m venv .venv
# Activate the environment
.\.venv\Scripts\Activate.ps1
# Install the package and test dependencies
python -m pip install -e ".[test]"
python -m pytest -q
```

NumPy, Numba, and the test dependencies are installed automatically. The first CEDI call includes Numba compilation. Run a warm-up call before measuring execution time.

## Four example in action

Run the examples from the project root after installation. They create vehicle states directly and do not require CSV files.

| Example                  | Command | Main output |
|--------------------------|---|---|
| RCD quick-start example  | `python examples/rcd_quick.py` | Conflict status and initial contact time |
| RCD step-by-step example | `python examples/rcd_step.py` | P1, P2, ET, ALT, and obstruction results |
| MLC conflict example     | `python examples/cedi_mlc.py` | CEDI, acceleration, waiting time, and interaction demands |
| Vehicle conflict example | `python examples/cedi_vehicle.py` | CEDI and action results for each vehicle |

See [Configuration](#configuration) for adjustable settings and [Inputs and outputs](#inputs-and-outputs) for data fields.

### RCD example

Vehicle C lies between A and B. RCD checks whether C blocks their predicted conflict.

```python
from rcd_cedi import VehicleState, detect_vehicle_conflict

a = VehicleState(1, 0, 0, 10, 0, 0, 4, 2)
b = VehicleState(2, 20, 0, 0, 0, 0, 4, 2)
c = VehicleState(3, 10, 0, 0, 1, 0, 4, 2)

result = detect_vehicle_conflict(a, b, surrounding=[c])
print(result.status)       # blocked
print(result.initial_et)   # 1.6 s
print(result.blockers)     # (3,)
```
RCD combines spatial interaction (P1) and conflict reachability (P2). A potentially blocking vehicle must clear before the predicted contact: `ALT_C < ET_AB`.
### MLC example

Define the work zone and evaluate the subject vehicle. Add other vehicles through `surrounding` when needed.

```python
from rcd_cedi import VehicleState, MLCConfig, evaluate_mlc

ego = VehicleState(10, -10, 1.75, 20, 0, 0, 4.8, 2.2)
config = MLCConfig(
    zone_start=(0.0, 0.0),
    zone_end=(100.0, 3.5),
    lane_width=3.5,
    lane_origin=1.75,
    road_min_d=0.0,
    road_max_d=10.5,
    tau_steps=30,
    ax_steps=30,
)
result = evaluate_mlc(ego, surrounding=[], config=config)
print(result.status, result.cedi)
print(result.ax, result.ay)
print(result.diagnostics.get("tau_s"))  # Selected waiting time, s
```

MLC considers the waiting time, the work-zone boundary, and relevant vehicles in the original and target lanes.
### Vehicle conflict example

Evaluate the same conflict from each vehicle's perspective, including the supplied surrounding vehicles.

```python
from rcd_cedi import VehicleState, VehicleConfig, evaluate_vehicle_pair

a = VehicleState(1, 0, 1.75, 10, 0, 0, 4, 2)
b = VehicleState(2, 20, 1.75, 0, 0, 0, 4, 2)
others = [VehicleState(3, -15, 1.75, 15, 0, 0, 4, 2)]
config = VehicleConfig(
    road_min=0.0, road_max=10.5, lane_width=3.5,
    ax_steps=30, ay_steps=30, sector_steps=30,
)
results = evaluate_vehicle_pair(a, b, others, config=config)
for vid, result in results.items():
    print(vid, result.status, result.cedi)
```
First, call `detect_vehicle_conflict()` to identify reachable conflicts. Then use `evaluate_vehicle_conflict()` to assess the subject vehicle’s risk, or `evaluate_vehicle_pair()` to assess both vehicles separately.

## Configuration

### Normalization scales and sensitivity

The denominators in the CEDI cost are **normalization scales**. They can be set to finite positive values according to the scenario and assessment objective to balance sensitivity to the different maneuvering demands. They do not have to match the paper's experimental settings.

For the same action and demand, a larger normalization scale reduces that term’s cost contribution, while a smaller scale increases it. In highway and work-zone scenarios where braking is preferred to lateral evasion, the scales can be set to assign a lower relative cost to braking.

| Model | Relevant configuration fields |
|---|---|
| Vehicle conflict | `ax_accel_g`, `ax_brake_g`, `ay_g`, `surround_brake_g` |
| MLC | `ax_accel_limits_g`, `ax_brake_limits_g`, `ay_limits_g`, `surrounding_brake_limits_g` |

The `_g` suffix denotes values in units of g, converted using `g = 9.81 m/s²`. For the MLC tuples, element `[0]` is used for cost normalization. The longitudinal acceleration and braking tuples also use element `[1]` in expanded search; it is not a second cost weight.

Normalization scales, physical constraints, and search resolution have different roles. Some scale settings also affect candidate grids or search ranges, so changing them can change both the cost and the available candidates.

### Where to change parameters

Defaults are defined in [`src/rcd_cedi/config.py`](src/rcd_cedi/config.py). Override them in your own script using `MLCConfig(...)` or `VehicleConfig(...)`; fields not supplied retain their defaults. You can also use `dataclasses.replace()` to create a modified configuration.

```python
from rcd_cedi import MLCConfig, VehicleConfig

mlc_config = MLCConfig(lane_width=3.5, tau_steps=30, ax_steps=30)
vehicle_config = VehicleConfig(
    road_min=0.0, road_max=10.5, lane_width=3.5,
    ax_steps=30, ay_steps=30, sector_steps=30,
    ax_brake_g=0.75, ay_g=0.30,
)
```

### Default settings

The defaults describe a generic example rather than a study site: three 3.5 m lanes with lateral bounds `[0, 10.5]` m and an MLC boundary from `(0, 0)` to `(100, 3.5)` m.

| Parameter | Default | Meaning |
|---|---|---|
| MLC `zone_start / zone_end` | `(0, 0) / (100, 3.5)` | Work-zone boundary endpoints, m; `zone_start[1]` must be zero |
| `lane_width` | 3.5 | Lane width, m |
| MLC `lane_origin` | 1.75 | Lateral coordinate of the lane 0 center, m |
| MLC `target_lane_offset` | 1 | Target lane ID minus original lane ID |
| MLC `road_min_d / road_max_d`; vehicle conflict `road_min / road_max` | 0 / 10.5 | Lateral road bounds, m |
| MLC `tau_steps / ax_steps` | 30 / 30 | Waiting-time and longitudinal search resolution |
| Vehicle conflict `ax_steps / ay_steps / sector_steps` | 30 / 30 / 30 | Longitudinal, lateral, and sector search resolution |
| MLC `ax_brake_limits_g`; vehicle conflict `ax_brake_g` | `(0.75, 1.50)`; 0.75 | Original braking settings; user-adjustable |
| Other acceleration scales, `gap`, timing, and expansion settings | See `config.py` | User-adjustable solver settings |

Lane centers, road bounds, work-zone endpoints, and vehicle positions are independent inputs. Check them together when changing the layout. All positions must use the same coordinate system.

Search resolution can affect the discrete optimum. MLC `ax_steps` also sets the finite expansion-stage budget; it is not a maximum physical acceleration. Vehicle conflict candidate counts also depend on additional grid points and sector sampling, so they are not simply `ax_steps × ay_steps`. Historical regression fixtures retain their explicit original settings and do not define the public defaults.

## Inputs and outputs

### Vehicle states and coordinates

```python
from rcd_cedi import VehicleState

ego = VehicleState(
    vid=1,
    x=0.0, y=1.75,        # m
    vx=10.0, vy=0.0,      # m/s
    heading=0.0,          # Body heading, rad
    length=4.0, width=2.0, # m
)
```

All moving actors are rectangles with fixed body headings during prediction. Body heading and velocity direction are stored separately and need not coincide. For MLC, `x/y/vx/vy` represent longitudinal/lateral position and velocity, with lanes aligned along the x-axis.

Vehicle IDs must be unique integers with absolute values no greater than `2**53`. Supply only the other vehicles in `surrounding`, excluding the subject and opponent. Synchronize trajectories and convert coordinates before calling the API.

### CSV fields

The API accepts `VehicleState` objects and does not include a built-in CSV loader or require a fixed CSV header. The following headers are recommended for mapping trajectory data to the API. Rename or convert columns explicitly if your dataset uses a different format.

| Recommended CSV field | API mapping | Unit / meaning |
|---|---|---|
| `time_s` | Used by the caller to group frames | s; build a scene from synchronized states |
| `vid` | `VehicleState.vid` | Unique integer vehicle ID |
| `x`, `y` | `x`, `y` | Rectangle center coordinates, m |
| `vx`, `vy` | `vx`, `vy` | Velocity components, m/s |
| `heading` | `heading` | Body heading, rad |
| `length`, `width` | `length`, `width` | Positive vehicle dimensions, m |

Example format:

```csv
time_s,vid,x,y,vx,vy,heading,length,width
0.0,1,0.0,1.75,10.0,0.0,0.0,4.0,2.0
0.0,2,20.0,1.75,0.0,0.0,0.0,4.0,2.0
```

Read and convert using the Python standard library:

```python
import csv
from rcd_cedi import VehicleState

frames = {}
with open("vehicle_states.csv", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        time_s = float(row["time_s"])
        state = VehicleState(
            vid=int(row["vid"]),
            **{key: float(row[key]) for key in
               ("x", "y", "vx", "vy", "heading", "length", "width")},
        )
        frames.setdefault(time_s, []).append(state)
```

This snippet only maps fields. Select the subject, opponent, and surrounding vehicles before evaluation. Align timestamps before grouping; do not mix states from different times. Road and work-zone geometry are supplied through configuration objects rather than repeated in every vehicle row.

### Result fields and CSV export

| Field | Result type | Meaning |
|---|---|---|
| `status` | RCD / CEDI | Detection or solver status |
| `subject_vehicle`, `opponent_vehicle` | RCD / CEDI | Vehicle IDs; the MLC opponent is `None` |
| `initial_et` | RCD / CEDI | Initial predicted contact time, s; may be `None` |
| `predicted_contact` | RCD | Whether contact is predicted |
| `reachable` | RCD | Whether the conflict is reachable; `None` if unresolved |
| `blockers`, `unresolved_vehicles` | RCD | Blocking vehicle IDs and unresolved initial-contact vehicle IDs |
| `cedi` | CEDI | Dimensionless cost; `None` for an unscored state |
| `ax`, `ay` | CEDI | Selected longitudinal/lateral acceleration, m/s² |
| `front_demand`, `rear_demand` | CEDI | Aggregated front-related and rear-related demands, m/s² |
| `responses` | CEDI | Surrounding-response details; MLC details are in `diagnostics` |
| `diagnostics` | RCD / CEDI | Geometry, search, timing, and audit details |

### CSV export

See [export_csv.py](examples/export_csv.py) for a runnable example and the reusable `save_results()` helper. It exports the main RCD or CEDI result fields; unavailable values are left blank.

Run `python examples/export_csv.py` to save an RCD example to `result.csv`. The same helper also accepts CEDI results, including both results returned by `evaluate_vehicle_pair()`.

### CEDI status values

`initial_et` always refers to the initial contact forecast. The MLC remaining ET after waiting is available as `diagnostics["ET_s"]`; the vehicle conflict action duration is `diagnostics["planning_duration_s"]`.

| Status | Meaning | CEDI |
|---|---|---|
| `feasible` | The numerical search found a feasible action | Numeric value |
| `no_initial_AB_conflict` | No initial constant-velocity contact predicted for the vehicle pair | 0 |
| `no_predicted_work_zone_contact` | No MLC boundary contact predicted | 0 |
| `initial_AB_contact` | The two vehicles are initially in contact | `None` |
| `already_at_work_zone` | The MLC vehicle is initially in contact with the boundary | `None` |
| `no_feasible_action_in_expanded_search` | No feasible action found in the final vehicle conflict candidate set | `None` |
| `no_feasible_action` | No MLC action found within the finite search budget | `None` |
| `initial_SV_work_zone_overlap` | The subject vehicle initially overlaps the static constraint | `None` |
| `initial_SV_road_boundary_violation` | The subject vehicle initially violates the road bounds | `None` |

Failure to find an action in a finite search is not proof of physical infeasibility in continuous control space. The package does not automatically write CSV or JSON files.
## Source code responsibilities

The modules below are located in `src/rcd_cedi/`.

| Module | Responsibility |
|---|---|
| `models.py` / `config.py` | Shared vehicle states, result objects, and model settings |
| `geometry.py` / `rcd.py` | Rectangle geometry, spatial interaction, and reachability |
| `motion.py` | Piecewise motion and continuous constraint checks |
| `static.py` | Static convex polygons, scene packing, and initial overlap checks |
| `search.py` / `responses.py` | Candidate actions and surrounding-vehicle responses |
| `cedi/mlc.py` / `cedi/vehicle.py` | MLC and vehicle conflict CEDI solvers |
| `audit.py` | Independent action checks and terminal-motion audits |
| `api.py` / `__init__.py` | Public interfaces |

## Citation



## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.