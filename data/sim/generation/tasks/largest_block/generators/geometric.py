"""Task-owned geometric factories for the largest-block Pick task."""

from data.sim.generation.tasks._pick import create_geometric_pick


def create_side_approach(context):
    return create_geometric_pick(
        context,
        generator_id="scripted_pick_side_approach_v1",
        profile="side_approach_v1",
    )


def create_yaw15(context):
    return create_geometric_pick(
        context,
        generator_id="scripted_pick_yaw15_v1",
        profile="yaw15_v1",
    )


def create_waypoint_lift(context):
    return create_geometric_pick(
        context,
        generator_id="scripted_pick_waypoint_lift_v1",
        profile="waypoint_lift_v1",
    )
