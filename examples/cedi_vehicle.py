"""Generic vehicle scene; all road and search settings can be overridden."""
from rcd_cedi import VehicleState, VehicleConfig, evaluate_vehicle_pair

config = VehicleConfig(
    road_min=0.0,
    road_max=10.5,
    lane_width=3.5,
    ax_steps=30,
    ay_steps=30,
    sector_steps=30,
)
lane_center = config.road_min + config.lane_width/2
a = VehicleState(1, 0, lane_center, 10, 0, 4, 2)
b = VehicleState(2, 20, lane_center, 0, 0, 4, 2)
others = [VehicleState(3, -15, lane_center, 15, 0, 4, 2),
          VehicleState(4, 22, lane_center+config.lane_width, 0, 0, 4, 2)]
for vid, result in evaluate_vehicle_pair(a, b, others, config).items():
    print(f'Vehicle {vid}: {result.status}, CEDI={result.cedi}')
    print('ax / ay:', result.ax, result.ay)
    print('front / rear demand:', result.front_demand, result.rear_demand)
