import math
import sys
from dataclasses import dataclass, field

import numpy as np

import ekf_cpp
import environment as env
import perception
import planner
from controller import WaypointController
from robot import DifferentialDriveRobot
from sensors import OdometrySensor

DT = 0.1
MAX_STEPS = 15000
INITIAL_COVARIANCE_DIAGONAL = 0.05

ROBOT_PHYSICAL_RADIUS = 0.2
COLLISION_BACKUP_SPEED = 0.6
COLLISION_BACKUP_MAX_TURN_RATE = 4.0
COLLISION_BACKUP_MIN_STEPS = 8
COLLISION_BACKUP_MAX_STEPS = 40
MAX_BACKUP_ANGLE_ATTEMPTS = 12


def _backup_arc_is_safe(start_pose: tuple, omega: float, n_steps: int) -> bool:
    x, y, theta = start_pose
    for _ in range(n_steps):
        x += -COLLISION_BACKUP_SPEED * math.cos(theta) * DT
        y += -COLLISION_BACKUP_SPEED * math.sin(theta) * DT
        theta += omega * DT
        if not env.is_point_free(x, y, margin=ROBOT_PHYSICAL_RADIUS):
            return False
    return True


@dataclass
class SimulationLog:
    planned_path: list
    true_poses: list = field(default_factory=list)
    ekf_estimates: list = field(default_factory=list)
    ekf_covariances: list = field(default_factory=list)
    dead_reckoning_poses: list = field(default_factory=list)
    detections_per_step: list = field(default_factory=list)
    goal_reached: bool = False
    steps_taken: int = 0


def run_simulation(odometry_seed: int = 0, perception_seed: int = 1) -> SimulationLog:
    landmarks_by_id = {lm.id: lm for lm in env.LANDMARKS}

    path = planner.plan_path(env.START_POSE[:2], env.GOAL_POSITION)
    if len(path) < 2:
        raise RuntimeError("A* failed to find a path from START_POSE to GOAL_POSITION")

    true_robot = DifferentialDriveRobot(*env.START_POSE)
    dead_reckoning_robot = DifferentialDriveRobot(*env.START_POSE)
    odometry = OdometrySensor(seed=odometry_seed)
    perception_rng = np.random.default_rng(perception_seed)
    bump_rng = np.random.default_rng(odometry_seed + 12345)
    controller = WaypointController()

    initial_state = np.array(env.START_POSE)
    initial_covariance = np.eye(3) * INITIAL_COVARIANCE_DIAGONAL
    ekf = ekf_cpp.EKF(initial_state, initial_covariance)

    log = SimulationLog(planned_path=path)
    waypoint_idx = 1
    backup_steps_remaining = 0
    backup_turn_rate = 0.0

    for step in range(MAX_STEPS):
        if step > 0 and step % 2000 == 0:
            print(f"  ...simulated {step}/{MAX_STEPS} steps so far")

        estimated_pose = tuple(ekf.get_state())
        target = path[waypoint_idx]

        was_backing_up = backup_steps_remaining > 0
        if was_backing_up:
            v, omega = -COLLISION_BACKUP_SPEED, backup_turn_rate
            backup_steps_remaining -= 1
        else:
            v, omega = controller.compute_control(estimated_pose, target, DT)

        candidate_x, candidate_y, _ = true_robot.predict_pose(v, omega, DT)
        if not env.is_point_free(candidate_x, candidate_y, margin=ROBOT_PHYSICAL_RADIUS):
            v, omega = 0.0, 0.0
            backup_steps_remaining = 0
            for _ in range(MAX_BACKUP_ANGLE_ATTEMPTS):
                candidate_turn = bump_rng.uniform(-COLLISION_BACKUP_MAX_TURN_RATE, COLLISION_BACKUP_MAX_TURN_RATE)
                candidate_steps = bump_rng.integers(COLLISION_BACKUP_MIN_STEPS, COLLISION_BACKUP_MAX_STEPS + 1)
                if _backup_arc_is_safe(true_robot.pose, candidate_turn, candidate_steps):
                    backup_turn_rate = candidate_turn
                    backup_steps_remaining = candidate_steps
                    v, omega = -COLLISION_BACKUP_SPEED, backup_turn_rate
                    break
            controller.reset()

        v_noisy, omega_noisy = odometry.sense(v, omega)
        ekf.predict(v_noisy, omega_noisy, DT)

        true_robot.step(v, omega, DT)
        dead_reckoning_robot.step(v_noisy, omega_noisy, DT)

        image = perception.render_camera_view(true_robot.pose, rng=perception_rng)
        detections = perception.detect_landmarks(image)
        for landmark_id, meas_range, meas_bearing in detections:
            landmark = landmarks_by_id[landmark_id]
            ekf.update(meas_range, meas_bearing, landmark.x, landmark.y)

        log.true_poses.append(true_robot.pose)
        log.ekf_estimates.append(tuple(ekf.get_state()))
        log.ekf_covariances.append(ekf.get_covariance())
        log.dead_reckoning_poses.append(dead_reckoning_robot.pose)
        log.detections_per_step.append(detections)

        if controller.has_reached(tuple(ekf.get_state()), target):
            controller.reset()
            waypoint_idx += 1
            if waypoint_idx >= len(path):
                log.goal_reached = True
                log.steps_taken = step + 1
                break
    else:
        log.steps_taken = MAX_STEPS

    return log


def scenario_from_argv(default: str = None) -> str:
    if "--scenario" in sys.argv:
        idx = sys.argv.index("--scenario")
        if idx + 1 >= len(sys.argv):
            raise SystemExit(f"--scenario needs a name, one of: {sorted(env.SCENARIOS)}")
        return sys.argv[idx + 1]
    return default or env.ACTIVE_SCENARIO


def seed_from_argv(default: int = 0) -> int:
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("--seed needs an integer")
        return int(sys.argv[idx + 1])
    return default


if __name__ == "__main__":
    env.load_scenario(scenario_from_argv())
    scenario = env.SCENARIOS[env.ACTIVE_SCENARIO]
    print(f"Scenario: {scenario.name} (difficulty {scenario.difficulty}/5): {scenario.description}")

    seed = seed_from_argv()
    result = run_simulation(odometry_seed=seed, perception_seed=seed + 1)
    status = "reached goal" if result.goal_reached else "did NOT reach goal (hit MAX_STEPS)"
    print(f"Simulation finished: {status} after {result.steps_taken} steps.")
    print(f"Planned path had {len(result.planned_path)} waypoints.")
    print(f"Final true pose: {result.true_poses[-1]}")
    print(f"Final EKF estimate: {result.ekf_estimates[-1]}")
    print(f"Final dead-reckoning pose: {result.dead_reckoning_poses[-1]}")
