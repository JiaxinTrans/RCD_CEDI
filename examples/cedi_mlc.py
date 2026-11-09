"""Generic MLC scene; edit config and vehicle states for your own road."""
from rcd_cedi import VehicleState, MLCConfig, evaluate_mlc

# These are editable example settings, not fixed model constants.
config = MLCConfig(
    zone_start=(0.0, 0.0),
    zone_end=(100.0, 3.5),
    lane_width=3.5,
    lane_origin=1.75,
    target_lane_offset=1,
    road_min_d=0.0,
    road_max_d=10.5,
    tau_steps=30,
    ax_steps=30,
)
original_center = config.lane_origin
target_center = original_center + config.target_lane_offset*config.lane_width
ego = VehicleState(10, -10, original_center, 20, 0, 0, 4.8, 2.2)
front = VehicleState(11, 20, target_center, 15, 0, 0, 4, 2)
rear = VehicleState(12, -30, target_center, 24, 0, 0, 4, 2)
result = evaluate_mlc(ego, [front, rear], config)
print('status:', result.status)
print('CEDI:', result.cedi)
print('ax / ay:', result.ax, result.ay)
print('waiting time:', result.diagnostics.get('tau_s'))
print('front / rear demand:', result.front_demand, result.rear_demand)
